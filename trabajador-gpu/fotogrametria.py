"""Fotogrametría de alto detalle guiada por ARKit + LiDAR (COLMAP + OpenMVS en Modal).

    modal run fotogrametria.py --carpeta "C:\\…\\Entorno_…"

1. Fotos clave (12 MP, reducidas a 2016 px) y fotogramas (1280 px) con su pose de ARKit.
2. COLMAP: puntos comunes entre fotos (SIFT en GPU), triangulación con las poses conocidas y ajuste de haces
   (corrige la deriva de ARKit: cada cámara queda donde de verdad estaba).
3. COLMAP: profundidad densa píxel a píxel (PatchMatch estéreo en GPU) y fusión en una nube densa.
4. OpenMVS: malla a partir de la nube densa y textura con nivelado global de costuras (sin parches de color).

Resultado en <carpeta>/fotogrametria/: malla texturizada (OBJ+MTL+PNG), nube densa y poses refinadas.
Las paredes lisas (sin textura) no salen en fotogrametría: ahí manda la malla LiDAR y el cascarón recreado.
"""

from __future__ import annotations

import json
import os
import shutil
import sqlite3
import subprocess
import uuid
from pathlib import Path

import modal

app = modal.App("escaner-lidar-fotogrametria")
datos = modal.Volume.from_name("escaner-lidar-fotogrametria", create_if_missing=True)

imagen_colmap = (modal.Image.from_registry("colmap/colmap:20260929.8468", add_python="3.11")
                 .pip_install("numpy==2.2.6"))
imagen_openmvs = modal.Image.from_registry("openmvs/openmvs-ubuntu:latest", add_python="3.11")


def ejecutar(cmd: list[str], cwd: Path | None = None):
    print("$", " ".join(cmd), flush=True)
    r = subprocess.run(cmd, cwd=cwd, capture_output=True, text=True)
    salida = (r.stdout + r.stderr).strip().splitlines()
    print("\n".join(salida[-15:]), flush=True)
    if r.returncode != 0:
        raise RuntimeError(f"{cmd[0]} {cmd[1] if len(cmd) > 1 else ''} falló ({r.returncode})")


# ----------------------------------------------------------------------------------------------- COLMAP (GPU)

@app.function(image=imagen_colmap, gpu="L4", volumes={"/datos": datos}, timeout=3 * 3600, cpu=8, memory=32768)
def colmap(trabajo: str) -> dict:
    import numpy as np

    datos.reload()
    raiz = Path("/datos") / trabajo
    imagenes, sparse, dense = raiz / "imagenes", raiz / "sparse", raiz / "dense"
    camaras = json.loads((raiz / "camaras.json").read_text())
    db = raiz / "database.db"

    # 1. Características SIFT (en GPU). Cada foto tiene su propia cámara (los intrínsecos varían un poco).
    ejecutar(["colmap", "feature_extractor", "--database_path", str(db), "--image_path", str(imagenes),
              "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "0",
              "--FeatureExtraction.use_gpu", "1", "--SiftExtraction.max_num_features", "8192"])

    # 2. Las poses e intrínsecos de ARKit se escriben con los mismos identificadores que la base de datos.
    con = sqlite3.connect(db)
    filas = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    conocida = sparse / "conocida"
    conocida.mkdir(parents=True, exist_ok=True)
    lineas_cam, lineas_img = [], []
    for image_id, nombre, camera_id in filas:
        c = camaras[nombre]
        params = np.array([c["fx"], c["fy"], c["cx"], c["cy"]], np.float64)
        con.execute("UPDATE cameras SET model=1, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
                    (c["ancho"], c["alto"], params.tobytes(), camera_id))
        lineas_cam.append(f"{camera_id} PINHOLE {c['ancho']} {c['alto']} {c['fx']} {c['fy']} {c['cx']} {c['cy']}")
        q, t = c["q"], c["t"]
        lineas_img.append(f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {camera_id} {nombre}\n")
    con.commit()
    con.close()
    (conocida / "cameras.txt").write_text("\n".join(lineas_cam) + "\n")
    (conocida / "images.txt").write_text("\n".join(lineas_img) + "\n")
    (conocida / "points3D.txt").write_text("")

    # 3. Emparejado y triangulación con las poses conocidas; luego ajuste de haces (intrínsecos fijos).
    ejecutar(["colmap", "exhaustive_matcher", "--database_path", str(db), "--FeatureMatching.use_gpu", "1"])
    tri, ba = sparse / "triangulada", sparse / "ajustada"
    tri.mkdir(exist_ok=True)
    ba.mkdir(exist_ok=True)
    ejecutar(["colmap", "point_triangulator", "--database_path", str(db), "--image_path", str(imagenes),
              "--input_path", str(conocida), "--output_path", str(tri)])
    ejecutar(["colmap", "bundle_adjuster", "--input_path", str(tri), "--output_path", str(ba),
              "--BundleAdjustment.refine_focal_length", "0", "--BundleAdjustment.refine_principal_point", "0",
              "--BundleAdjustment.refine_extra_params", "0"])
    ejecutar(["colmap", "model_converter", "--input_path", str(ba), "--output_path", str(ba), "--output_type", "TXT"])

    # 4. Profundidad densa (PatchMatch en GPU) y fusión.
    ejecutar(["colmap", "image_undistorter", "--image_path", str(imagenes), "--input_path", str(ba),
              "--output_path", str(dense), "--output_type", "COLMAP", "--max_image_size", "2000"])
    ejecutar(["colmap", "patch_match_stereo", "--workspace_path", str(dense),
              "--PatchMatchStereo.geom_consistency", "true", "--PatchMatchStereo.max_image_size", "1600"])
    ejecutar(["colmap", "model_converter", "--input_path", str(dense / "sparse"), "--output_path", str(dense / "sparse"),
              "--output_type", "TXT"])
    ejecutar(["colmap", "stereo_fusion", "--workspace_path", str(dense), "--output_path", str(dense / "fused.ply"),
              "--StereoFusion.min_num_pixels", "3"])
    datos.commit()

    puntos = sum(1 for linea in (ba / "points3D.txt").read_text().splitlines() if linea and not linea.startswith("#"))
    return {"imagenes": len(filas), "puntos_dispersos": puntos,
            "nube_densa_MB": round((dense / "fused.ply").stat().st_size / 1e6, 1)}


# ----------------------------------------------------------------------------------------------- OpenMVS (CPU)

@app.function(image=imagen_openmvs, volumes={"/datos": datos}, timeout=3 * 3600, cpu=16, memory=65536)
def openmvs(trabajo: str) -> dict:
    datos.reload()
    raiz = Path("/datos") / trabajo
    dense, mvs = raiz / "dense", raiz / "mvs"
    mvs.mkdir(exist_ok=True)
    binario = lambda n: shutil.which(n) or next(str(p) for p in Path("/").glob(f"**/bin/**/{n}") if p.is_file())
    ejecutar([binario("InterfaceCOLMAP"), "-i", str(dense), "-o", str(mvs / "escena.mvs"),
              "--image-folder", str(dense / "images")], cwd=mvs)
    ejecutar([binario("ReconstructMesh"), "escena.mvs", "-p", str(dense / "fused.ply"), "-o", "malla.mvs"], cwd=mvs)
    ejecutar([binario("RefineMesh"), "malla.mvs", "-m", "malla.ply", "--resolution-level", "1",
              "--scales", "2", "-o", "malla_refinada.mvs"], cwd=mvs)
    ejecutar([binario("TextureMesh"), "malla.mvs", "-m", "malla_refinada.ply", "--export-type", "obj",
              "--resolution-level", "0", "-o", "texturizada.mvs"], cwd=mvs)
    datos.commit()
    return {"archivos": sorted(p.name for p in mvs.iterdir())}


# ----------------------------------------------------------------------------------------------- en el PC

@app.local_entrypoint()
def principal(carpeta: str, max_fotogramas: int = 160):
    import numpy as np
    from PIL import Image

    origen = Path(carpeta).expanduser().resolve()
    trabajo = f"{origen.name[:50]}-{uuid.uuid4().hex[:6]}"
    local = Path(os.environ.get("TEMP", "/tmp")) / "fotogrametria" / trabajo
    (local / "imagenes").mkdir(parents=True, exist_ok=True)
    arkit_a_opencv = np.diag([1.0, -1.0, -1.0, 1.0])
    camaras = {}

    def cuaternion(r):
        w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
        x = np.copysign(np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
        y = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
        z = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
        return [w, x, y, z]

    for sub, lado, limite in (("fotos_clave", 2016, 10_000), ("fotogramas", 1280, max_fotogramas)):
        info = json.loads((origen / sub / "camaras.json").read_text("utf-8"))["fotos"]
        paso = max(1, len(info) // limite)
        for f in info[::paso][:limite]:
            img = Image.open(origen / sub / f["archivo"])
            s = min(1.0, lado / max(img.size))
            if s < 1:
                img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
            sx, sy = img.width / f["ancho"], img.height / f["alto"]
            nombre = f"{sub[:3]}_{Path(f['archivo']).stem}.jpg"
            img.convert("RGB").save(local / "imagenes" / nombre, quality=95)
            c2w = np.asarray(f["cameraToWorld"], float).reshape(4, 4).T
            w2c = arkit_a_opencv @ np.linalg.inv(c2w)
            camaras[nombre] = {"ancho": img.width, "alto": img.height, "fx": f["fx"] * sx, "fy": f["fy"] * sy,
                               "cx": (f["cx"] + 0.5) * sx - 0.5, "cy": (f["cy"] + 0.5) * sy - 0.5,
                               "q": cuaternion(w2c[:3, :3]), "t": w2c[:3, 3].tolist()}
    (local / "camaras.json").write_text(json.dumps(camaras))
    print(f"{len(camaras)} imágenes preparadas; subiendo…")
    with datos.batch_upload(force=True) as lote:
        lote.put_directory(str(local), f"/{trabajo}")

    print("COLMAP:", colmap.remote(trabajo))
    print("OpenMVS:", openmvs.remote(trabajo))

    destino = origen / "fotogrametria"
    destino.mkdir(exist_ok=True)
    for ruta in ("mvs/texturizada.obj", "mvs/texturizada.mtl", "dense/fused.ply"):
        try:
            with open(destino / Path(ruta).name, "wb") as fh:
                for trozo in datos.read_file(f"/{trabajo}/{ruta}"):
                    fh.write(trozo)
        except Exception as error:
            print("no se pudo bajar", ruta, error)
    for entrada in datos.listdir(f"/{trabajo}/mvs"):
        if entrada.path.endswith((".png", ".jpg")) and "texturizada" in entrada.path:
            with open(destino / Path(entrada.path).name, "wb") as fh:
                for trozo in datos.read_file(entrada.path):
                    fh.write(trozo)
    for nombre in ("images.txt", "points3D.txt"):
        with open(destino / f"poses_{nombre}", "wb") as fh:
            for trozo in datos.read_file(f"/{trabajo}/sparse/ajustada/{nombre}"):
                fh.write(trozo)
    print("Listo en", destino)
