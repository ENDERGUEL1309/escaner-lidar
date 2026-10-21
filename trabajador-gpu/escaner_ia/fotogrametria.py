"""Fotogrametría de alto detalle guiada por ARKit + LiDAR: los mismos pasos en el PC y en la nube.

1. preparar: fotos clave (12 MP → 2016 px) y fotogramas (1280 px) con su pose de ARKit en formato COLMAP.
2. colmap: SIFT en GPU, triangulación con las poses conocidas, ajuste de haces (corrige la deriva de ARKit),
   profundidad densa por PatchMatch en GPU y fusión en una nube densa.
3. openmvs: malla a partir de la nube densa, refinado fotométrico y textura con nivelado global de costuras.

`colmap_cmd` es la orden con la que se llama a COLMAP (en Windows: ["cmd", "/c", r"…\\COLMAP.bat"]) y
`openmvs_bin` la carpeta con los ejecutables de OpenMVS (o None si están en el PATH).
"""

from __future__ import annotations

import json
import shutil
import sqlite3
import subprocess
import time
from pathlib import Path

import numpy as np

ARKIT_A_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


def ejecutar(cmd: list[str], cwd: Path | None = None, registro=print):
    registro("$ " + " ".join(str(c) for c in cmd))
    t0 = time.time()
    r = subprocess.run([str(c) for c in cmd], cwd=cwd, capture_output=True, text=True, errors="replace")
    salida = (r.stdout + r.stderr).strip().splitlines()
    registro("\n".join(salida[-8:]))
    registro(f"  ({(time.time() - t0) / 60:.1f} min)")
    if r.returncode != 0:
        raise RuntimeError(f"{Path(str(cmd[-1] if cmd[0] == 'cmd' else cmd[0])).name} falló ({r.returncode}): "
                           + "\n".join(salida[-20:]))


def _cuaternion(r):
    w = np.sqrt(max(0.0, 1 + r[0, 0] + r[1, 1] + r[2, 2])) / 2
    x = np.copysign(np.sqrt(max(0.0, 1 + r[0, 0] - r[1, 1] - r[2, 2])) / 2, r[2, 1] - r[1, 2])
    y = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] + r[1, 1] - r[2, 2])) / 2, r[0, 2] - r[2, 0])
    z = np.copysign(np.sqrt(max(0.0, 1 - r[0, 0] - r[1, 1] + r[2, 2])) / 2, r[1, 0] - r[0, 1])
    return [float(w), float(x), float(y), float(z)]


def _rotacion(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def similitud_arkit(images_txt: Path, camaras: dict) -> np.ndarray:
    """Similitud 4x4 (Umeyama) que lleva los centros de cámara de un modelo COLMAP a los de ARKit."""
    a, b = [], []
    lineas = [l for l in images_txt.read_text().splitlines() if l.strip() and not l.startswith("#")]
    for linea in lineas[::2]:
        p = linea.split()
        r, t = _rotacion(list(map(float, p[1:5]))), np.array(list(map(float, p[5:8])))
        k = camaras[p[9]]
        a.append(-r.T @ t)
        b.append(-_rotacion(k["q"]).T @ np.array(k["t"]))
    a, b = np.array(a), np.array(b)
    ma, mb = a.mean(0), b.mean(0)
    u, d, vt = np.linalg.svd((b - mb).T @ (a - ma) / len(a))
    sg = np.diag([1.0, 1.0, np.sign(np.linalg.det(u @ vt))])
    r = u @ sg @ vt
    s = np.trace(np.diag(d) @ sg) / ((a - ma) ** 2).sum(1).mean()
    sim = np.eye(4)
    sim[:3, :3], sim[:3, 3] = s * r, mb - s * r @ ma
    return sim


def preparar(origen: Path, destino: Path, max_fotogramas: int = 160) -> int:
    """Copia las imágenes reducidas a destino/imagenes y escribe destino/camaras.json (pose en convención OpenCV)."""
    from PIL import Image

    (destino / "imagenes").mkdir(parents=True, exist_ok=True)
    camaras = {}
    for sub, lado, limite in (("fotos_clave", 2016, 10_000), ("fotogramas", 1280, max_fotogramas)):
        ruta = origen / sub / "camaras.json"
        if not ruta.exists():
            continue
        info = json.loads(ruta.read_text("utf-8"))["fotos"]
        paso = max(1, len(info) // max(1, limite))
        for f in info[::paso][:limite]:
            img = Image.open(origen / sub / f["archivo"])
            s = min(1.0, lado / max(img.size))
            if s < 1:
                img = img.resize((round(img.width * s), round(img.height * s)), Image.LANCZOS)
            sx, sy = img.width / f["ancho"], img.height / f["alto"]
            nombre = f"{sub[:3]}_{Path(f['archivo']).stem}.jpg"
            img.convert("RGB").save(destino / "imagenes" / nombre, quality=95)
            c2w = np.asarray(f["cameraToWorld"], float).reshape(4, 4).T
            w2c = ARKIT_A_OPENCV @ np.linalg.inv(c2w)
            camaras[nombre] = {"ancho": img.width, "alto": img.height, "fx": f["fx"] * sx, "fy": f["fy"] * sy,
                               "cx": (f["cx"] + 0.5) * sx - 0.5, "cy": (f["cy"] + 0.5) * sy - 0.5,
                               "q": _cuaternion(w2c[:3, :3]), "t": w2c[:3, 3].tolist()}
    (destino / "camaras.json").write_text(json.dumps(camaras))
    return len(camaras)


def colmap(raiz: Path, colmap_cmd: list[str], registro=print, max_imagen: int = 2000) -> dict:
    imagenes, sparse, dense = raiz / "imagenes", raiz / "sparse", raiz / "dense"
    camaras = json.loads((raiz / "camaras.json").read_text())
    db = raiz / "database.db"
    c = lambda *args: ejecutar([*colmap_cmd, *args], registro=registro)

    if not db.exists():
        c("feature_extractor", "--database_path", db, "--image_path", imagenes,
          "--ImageReader.camera_model", "PINHOLE", "--ImageReader.single_camera", "0",
          "--FeatureExtraction.use_gpu", "1", "--SiftExtraction.max_num_features", "8192")

    # Poses e intrínsecos de ARKit con los mismos identificadores que la base de datos de COLMAP.
    con = sqlite3.connect(db)
    filas = con.execute("SELECT image_id, name, camera_id FROM images").fetchall()
    conocida = sparse / "conocida"
    conocida.mkdir(parents=True, exist_ok=True)
    cams, imgs = [], []
    for image_id, nombre, camera_id in filas:
        k = camaras[nombre]
        params = np.array([k["fx"], k["fy"], k["cx"], k["cy"]], np.float64)
        con.execute("UPDATE cameras SET model=1, width=?, height=?, params=?, prior_focal_length=1 WHERE camera_id=?",
                    (k["ancho"], k["alto"], params.tobytes(), camera_id))
        cams.append(f"{camera_id} PINHOLE {k['ancho']} {k['alto']} {k['fx']} {k['fy']} {k['cx']} {k['cy']}")
        q, t = k["q"], k["t"]
        imgs.append(f"{image_id} {q[0]} {q[1]} {q[2]} {q[3]} {t[0]} {t[1]} {t[2]} {camera_id} {nombre}\n")
    con.commit()
    con.close()
    (conocida / "cameras.txt").write_text("\n".join(cams) + "\n")
    (conocida / "images.txt").write_text("\n".join(imgs) + "\n")
    (conocida / "points3D.txt").write_text("")

    c("exhaustive_matcher", "--database_path", db, "--FeatureMatching.use_gpu", "1")
    tri, ba = sparse / "triangulada", sparse / "ajustada"
    tri.mkdir(exist_ok=True)
    ba.mkdir(exist_ok=True)
    c("point_triangulator", "--database_path", db, "--image_path", imagenes, "--input_path", conocida, "--output_path", tri)
    c("bundle_adjuster", "--input_path", tri, "--output_path", ba, "--BundleAdjustment.refine_focal_length", "0",
      "--BundleAdjustment.refine_principal_point", "0", "--BundleAdjustment.refine_extra_params", "0")
    c("model_converter", "--input_path", ba, "--output_path", ba, "--output_type", "TXT")
    # El ajuste de haces deja libre la escala (aquí llegó a duplicarla): se devuelve el modelo a metros y al mundo
    # de ARKit con la similitud que mejor lleva los centros de cámara ajustados a los de ARKit.
    sim = similitud_arkit(ba / "images.txt", camaras)
    escala = np.cbrt(np.linalg.det(sim[:3, :3]))
    registro(f"  escala tras el ajuste de haces: {1 / escala:.3f} → corregida")
    # Formato Sim3d de COLMAP: escala qw qx qy qz tx ty tz
    (sparse / "a_arkit.txt").write_text(" ".join(f"{x:.12f}" for x in
                                                  [escala, *_cuaternion(sim[:3, :3] / escala), *sim[:3, 3]]) + "\n")
    c("model_transformer", "--input_path", ba, "--output_path", ba, "--transform_path", sparse / "a_arkit.txt")
    c("model_converter", "--input_path", ba, "--output_path", ba, "--output_type", "TXT")
    c("image_undistorter", "--image_path", imagenes, "--input_path", ba, "--output_path", dense,
      "--output_type", "COLMAP", "--max_image_size", str(max_imagen))
    c("patch_match_stereo", "--workspace_path", dense, "--PatchMatchStereo.geom_consistency", "true",
      "--PatchMatchStereo.max_image_size", "1600")
    c("stereo_fusion", "--workspace_path", dense, "--output_path", dense / "fused.ply", "--StereoFusion.min_num_pixels", "3")

    puntos = sum(1 for linea in (ba / "points3D.txt").read_text().splitlines() if linea and not linea.startswith("#"))
    return {"imagenes": len(filas), "puntos_dispersos": puntos,
            "nube_densa_MB": round((dense / "fused.ply").stat().st_size / 1e6, 1)}


def openmvs(raiz: Path, openmvs_bin: Path | None, registro=print, colmap_cmd: list[str] | None = None) -> dict:
    dense, mvs = raiz / "dense", raiz / "mvs"
    mvs.mkdir(exist_ok=True)
    # image_undistorter guarda las cámaras en binario; InterfaceCOLMAP (sobre todo en versiones antiguas) lee texto.
    if colmap_cmd and not (dense / "sparse" / "cameras.txt").exists():
        ejecutar([*colmap_cmd, "model_converter", "--input_path", dense / "sparse", "--output_path", dense / "sparse",
                  "--output_type", "TXT"], registro=registro)
    def b(nombre):
        if openmvs_bin:
            return Path(openmvs_bin) / (nombre + (".exe" if (Path(openmvs_bin) / (nombre + ".exe")).exists() else ""))
        return shutil.which(nombre) or next(str(p) for p in Path("/").glob(f"**/bin/**/{nombre}") if p.is_file())
    e = lambda *args: ejecutar(list(args), cwd=mvs, registro=registro)
    # Cada paso se salta si su resultado ya existe, para poder reanudar.
    if not (mvs / "escena.ply").exists():
        e(b("InterfaceCOLMAP"), "-i", dense, "-o", mvs / "escena.mvs", "--image-folder", dense / "images")
    # escena.ply es la nube densa con la visibilidad por cámara; fused.ply sola hace fallar a ReconstructMesh.
    if not (mvs / "malla.ply").exists():
        e(b("ReconstructMesh"), "escena.mvs", "-p", "escena.ply", "-o", "malla.mvs")
    if not (mvs / "malla_refinada.ply").exists():
        e(b("RefineMesh"), "escena.mvs", "-m", "malla.ply", "--resolution-level", "1", "--scales", "2",
          "--decimate", "1", "-o", "malla_refinada.mvs")  # sin reducir: conserva todo el detalle
    e(b("TextureMesh"), "escena.mvs", "-m", "malla_refinada.ply", "--export-type", "obj", "--resolution-level", "0",
      "-o", "texturizada.mvs")
    return {"archivos": sorted(p.name for p in mvs.iterdir())}
