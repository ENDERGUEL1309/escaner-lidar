"""Gaussian Splatting 2D (2DGS) guiado por LiDAR en este PC, y extracción de una malla limpia.

    .venv-splat\\Scripts\\python splat2d_local.py "C:\\…\\Entorno_…"

Necesita haber pasado antes por fotogrametria_local.py (usa sus imágenes y las poses del ajuste de haces).
Cada gaussiana es un disco plano (2DGS), así que las superficies salen nítidas y la malla se puede extraer
directamente de la profundidad renderizada. La profundidad de la malla LiDAR supervisa la geometría:
las paredes lisas y blancas, que la fotogrametría no ve, quedan planas y en su sitio.

Resultado en <escaneo>/splat2d/: malla_2dgs.ply (color por vértice, mundo ARKit con Y arriba, en metros)
y splats.ply (para visores de Gaussian Splatting).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
HERRAMIENTAS = Path.home() / "Herramientas"


# ----------------------------------------------------------------------------------------------- datos

def _rotacion(q):
    w, x, y, z = q
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - w * z), 2 * (x * z + w * y)],
                     [2 * (x * y + w * z), 1 - 2 * (x * x + z * z), 2 * (y * z - w * x)],
                     [2 * (x * z - w * y), 2 * (y * z + w * x), 1 - 2 * (x * x + y * y)]])


def leer_ply(ruta: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vértices y colores (0-1) de un PLY binario con x y z … red green blue."""
    crudo = ruta.read_bytes()
    fin = crudo.index(b"\n", crudo.index(b"end_header")) + 1
    cabecera = crudo[:fin].decode(errors="replace").splitlines()
    n = next(int(l.split()[-1]) for l in cabecera if l.startswith("element vertex"))
    tipos = {"float": "f4", "uchar": "u1", "double": "f8", "int": "i4", "uint": "u4"}
    props = []
    for l in cabecera:
        if l.startswith("element") and not l.startswith("element vertex") and props:
            break
        if l.startswith("property") and "list" not in l:
            _, t, nombre = l.split()
            props.append((nombre, tipos[t]))
    v = np.frombuffer(crudo[fin:fin + n * np.dtype(props).itemsize], np.dtype(props))
    xyz = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    rgb = np.stack([v["red"], v["green"], v["blue"]], 1).astype(np.float32) / 255
    return xyz, rgb


def cargar_vistas(trabajo: Path, T: np.ndarray, lado: int):
    """Imágenes, intrínsecos y poses (mundo a cámara, convención OpenCV) en el mundo LiDAR."""
    camaras = json.loads((trabajo / "camaras.json").read_text())
    s = np.cbrt(np.linalg.det(T[:3, :3]))
    RT, tT = T[:3, :3] / s, T[:3, 3]
    lineas = [l for l in (trabajo / "sparse" / "ajustada" / "images.txt").read_text().splitlines()
              if l.strip() and not l.startswith("#")][::2]
    vistas = []
    for linea in lineas:
        p = linea.split()
        R = _rotacion(list(map(float, p[1:5])))
        t = np.array(list(map(float, p[5:8])))
        centro = s * RT @ (-R.T @ t) + tT
        Rc2w = RT @ R.T
        w2c = np.eye(4)
        w2c[:3, :3], w2c[:3, 3] = Rc2w.T, -Rc2w.T @ centro
        k = camaras[p[9]]
        img = Image.open(trabajo / "imagenes" / p[9]).convert("RGB")
        f = min(1.0, lado / max(img.size))
        if f < 1:
            img = img.resize((round(img.width * f), round(img.height * f)), Image.LANCZOS)
        fx, fy = img.width / k["ancho"], img.height / k["alto"]
        K = np.array([[k["fx"] * fx, 0, (k["cx"] + 0.5) * fx - 0.5],
                      [0, k["fy"] * fy, (k["cy"] + 0.5) * fy - 0.5], [0, 0, 1]], np.float32)
        vistas.append({"nombre": p[9], "img": torch.from_numpy(np.asarray(img).copy()), "K": K,
                       "w2c": w2c.astype(np.float32), "ancho": img.width, "alto": img.height})
    return vistas


@torch.no_grad()
def profundidad_lidar(vertices: torch.Tensor, w2c: np.ndarray, K: np.ndarray, ancho: int, alto: int, red: int = 4):
    """Proyecta la malla LiDAR (z-buffer por puntos) a 1/red de resolución y rellena huecos pequeños."""
    W, H = ancho // red, alto // red
    M = torch.from_numpy(w2c).to(vertices)
    p = vertices @ M[:3, :3].T + M[:3, 3]
    p = p[p[:, 2] > 0.05]
    u = (p[:, 0] / p[:, 2] * K[0, 0] + K[0, 2]) / red
    v = (p[:, 1] / p[:, 2] * K[1, 1] + K[1, 2]) / red
    ok = (u >= 0) & (u < W) & (v >= 0) & (v < H)
    idx = v[ok].long() * W + u[ok].long()
    prof = torch.full((H * W,), float("inf"), device=vertices.device)
    prof.scatter_reduce_(0, idx, p[ok, 2], reduce="amin")
    prof = prof.view(1, 1, H, W)
    # Huecos de un píxel (entre vértices): mínimo del vecindario, solo donde falta el dato.
    vecino = -F.max_pool2d(-prof.nan_to_num(posinf=1e3), 3, 1, 1)
    prof = torch.where(torch.isinf(prof) & (vecino < 1e3), vecino, prof)
    prof[torch.isinf(prof)] = 0
    return prof[0, 0]


def alinear_con_lidar(origen: Path, trabajo: Path, ruta: Path, registro=print) -> np.ndarray:
    """Transformación del modelo COLMAP al mundo de la malla LiDAR: similitud que lleva las cámaras ajustadas a las
    de ARKit (corrige la escala libre del ajuste de haces) y ICP de la nube densa contra la malla LiDAR (corrige los
    centímetros y grados de diferencia entre las poses de ARKit y la fusión de la app)."""
    from scipy.spatial import cKDTree

    from escaner_ia.fotogrametria import similitud_arkit

    camaras = json.loads((trabajo / "camaras.json").read_text())
    T = similitud_arkit(trabajo / "sparse" / "ajustada" / "images.txt", camaras)
    densa, _ = leer_ply(trabajo / "dense" / "fused.ply")
    lidar, _ = leer_ply(origen / "malla_color.ply")
    rng = np.random.default_rng(0)
    fuente = densa[rng.choice(len(densa), min(60_000, len(densa)), replace=False)].astype(np.float64)
    fuente = fuente @ T[:3, :3].T + T[:3, 3]
    arbol = cKDTree(lidar[rng.choice(len(lidar), min(500_000, len(lidar)), replace=False)])
    d0 = arbol.query(fuente)[0]
    R, t = np.eye(3), np.zeros(3)
    for _ in range(40):
        p = fuente @ R.T + t
        d, idx = arbol.query(p)
        sel = d < max(0.02, np.percentile(d, 60))
        a, b = p[sel], arbol.data[idx[sel]]
        ma, mb = a.mean(0), b.mean(0)
        U, _, Vt = np.linalg.svd((a - ma).T @ (b - mb))
        Ri = Vt.T @ np.diag([1, 1, np.sign(np.linalg.det(Vt.T @ U.T))]) @ U.T
        R, t = Ri @ R, Ri @ t + mb - Ri @ ma
    d1 = arbol.query(fuente @ R.T + t)[0]
    C = np.eye(4)
    C[:3, :3], C[:3, 3] = R, t
    T = C @ T
    escala = np.cbrt(np.linalg.det(T[:3, :3]))
    registro(f"Alineación con el LiDAR: escala {escala:.4f}, giro ICP {np.degrees(np.arccos(min(1, (np.trace(R) - 1) / 2))):.2f}°, "
             f"distancia mediana {np.median(d0) * 100:.1f} → {np.median(d1) * 100:.1f} cm")
    ruta.parent.mkdir(exist_ok=True)
    ruta.write_text(json.dumps({"descripcion": "similitud 4x4: modelo COLMAP tras el ajuste de haces -> mundo "
                                               "ARKit/LiDAR (Y arriba)", "matriz": T.tolist()}, indent=1))
    return T


# ----------------------------------------------------------------------------------------------- utilidades

def ssim(a: torch.Tensor, b: torch.Tensor) -> torch.Tensor:
    """SSIM con ventana gaussiana 11x11; a y b en (1, 3, H, W)."""
    g = torch.exp(-((torch.arange(11, device=a.device) - 5) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    w = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11)
    mu_a, mu_b = F.conv2d(a, w, padding=5, groups=3), F.conv2d(b, w, padding=5, groups=3)
    va = F.conv2d(a * a, w, padding=5, groups=3) - mu_a ** 2
    vb = F.conv2d(b * b, w, padding=5, groups=3) - mu_b ** 2
    cov = F.conv2d(a * b, w, padding=5, groups=3) - mu_a * mu_b
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mu_a * mu_b + c1) * (2 * cov + c2)) / ((mu_a ** 2 + mu_b ** 2 + c1) * (va + vb + c2))).mean()


def rgb_a_sh(rgb):
    return (rgb - 0.5) / 0.28209479177387814


def guardar_splats(ruta: Path, splats: dict):
    """PLY en el formato estándar de 3DGS (los discos 2DGS llevan la tercera escala muy pequeña)."""
    medias = splats["means"].detach().cpu().numpy()
    n = len(medias)
    sh0 = splats["sh0"].detach().transpose(1, 2).reshape(n, -1).cpu().numpy()
    shn = splats["shN"].detach().transpose(1, 2).reshape(n, -1).cpu().numpy()
    escalas = splats["scales"].detach().clone()
    escalas[:, 2] = math.log(1e-4)
    campos = ["x", "y", "z", "nx", "ny", "nz"] + [f"f_dc_{i}" for i in range(sh0.shape[1])] + \
             [f"f_rest_{i}" for i in range(shn.shape[1])] + ["opacity", "scale_0", "scale_1", "scale_2",
                                                              "rot_0", "rot_1", "rot_2", "rot_3"]
    datos = np.concatenate([medias, np.zeros_like(medias), sh0, shn,
                            splats["opacities"].detach().cpu().numpy()[:, None],
                            escalas.cpu().numpy(), splats["quats"].detach().cpu().numpy()], 1).astype(np.float32)
    with open(ruta, "wb") as fh:
        fh.write(("ply\nformat binary_little_endian 1.0\n" + f"element vertex {n}\n" +
                  "".join(f"property float {c}\n" for c in campos) + "end_header\n").encode())
        fh.write(datos.tobytes())


# ----------------------------------------------------------------------------------------------- principal

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    p.add_argument("--pasos", type=int, default=20_000)
    p.add_argument("--umbral-crecimiento", type=float, default=0.0006,
                   help="gradiente 2D a partir del cual una gaussiana se divide (más alto = menos gaussianas)")
    p.add_argument("--lado", type=int, default=1280, help="lado mayor de las imágenes de entrenamiento")
    p.add_argument("--peso-profundidad", type=float, default=0.3)
    p.add_argument("--voxel", type=float, default=0.008, help="tamaño de vóxel de la fusión TSDF (m)")
    p.add_argument("--solo-malla", action="store_true", help="no entrenar: extraer la malla de splats.pt")
    a = p.parse_args()

    from gsplat import rasterization_2dgs
    from gsplat.strategy import DefaultStrategy

    origen = Path(a.carpeta).resolve()
    trabajo = HERRAMIENTAS / "trabajos" / origen.name
    destino = origen / "splat2d"
    destino.mkdir(exist_ok=True)
    registro_txt = open(destino / "registro.txt", "a", encoding="utf-8")

    def registro(texto):
        print(texto, flush=True)
        registro_txt.write(texto + "\n")
        registro_txt.flush()

    dev = torch.device("cuda")
    ruta_T = origen / "fotogrametria" / "colmap_a_lidar.json"
    if not ruta_T.exists():
        alinear_con_lidar(origen, trabajo, ruta_T, registro)
    T = np.array(json.loads(ruta_T.read_text())["matriz"])
    vistas = cargar_vistas(trabajo, T, a.lado)
    registro(f"{len(vistas)} vistas a {vistas[0]['ancho']}x{vistas[0]['alto']} (fotos clave) y "
             f"{vistas[-1]['ancho']}x{vistas[-1]['alto']} (fotogramas)")

    lidar_xyz, lidar_rgb = leer_ply(origen / "malla_color.ply")
    lidar = torch.from_numpy(lidar_xyz).to(dev)
    t0 = time.time()
    for v in vistas:
        v["prof"] = profundidad_lidar(lidar, v["w2c"], v["K"], v["ancho"], v["alto"]).cpu()
    cobertura = np.mean([(v["prof"] > 0).float().mean().item() for v in vistas])
    registro(f"Profundidad LiDAR proyectada en {time.time() - t0:.0f} s (cobertura media {cobertura:.0%})")

    centros = np.array([np.linalg.inv(v["w2c"])[:3, 3] for v in vistas])
    escena = float(1.1 * np.linalg.norm(centros - centros.mean(0), axis=1).max())

    if not a.solo_malla:
        # Inicialización: nube densa de COLMAP (detalle donde hay textura) + malla LiDAR (paredes lisas).
        densa_xyz, densa_rgb = leer_ply(trabajo / "dense" / "fused.ply")
        densa_xyz = densa_xyz @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
        rng = np.random.default_rng(0)
        i1 = rng.choice(len(densa_xyz), min(400_000, len(densa_xyz)), replace=False)
        i2 = rng.choice(len(lidar_xyz), min(200_000, len(lidar_xyz)), replace=False)
        xyz = torch.from_numpy(np.concatenate([densa_xyz[i1], lidar_xyz[i2]])).to(dev)
        rgb = torch.from_numpy(np.concatenate([densa_rgb[i1], lidar_rgb[i2]])).to(dev)
        # Escala inicial: distancia media a los 3 vecinos más próximos (por bloques para no agotar memoria).
        d = torch.empty(len(xyz), device=dev)
        for i in range(0, len(xyz), 512):
            dist = torch.cdist(xyz[i:i + 512], xyz)
            d[i:i + 512] = dist.topk(4, largest=False).values[:, 1:].mean(1)
        n = len(xyz)
        grado = 3
        splats = torch.nn.ParameterDict({
            "means": torch.nn.Parameter(xyz),
            "scales": torch.nn.Parameter(torch.log(d.clamp(1e-3, 0.05))[:, None].repeat(1, 3)),
            "quats": torch.nn.Parameter(F.normalize(torch.rand(n, 4, device=dev), dim=-1)),
            "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.5, device=dev))),
            "sh0": torch.nn.Parameter(rgb_a_sh(rgb)[:, None, :]),
            "shN": torch.nn.Parameter(torch.zeros(n, (grado + 1) ** 2 - 1, 3, device=dev)),
        })
        tasas = {"means": 1.6e-4 * escena, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2,
                 "sh0": 2.5e-3, "shN": 2.5e-3 / 20}
        optim = {k: torch.optim.Adam([{"params": splats[k], "lr": tasas[k], "name": k}], eps=1e-15)
                 for k in splats}
        decaimiento = torch.optim.lr_scheduler.ExponentialLR(optim["means"], gamma=0.01 ** (1 / a.pasos))
        # Corrección de exposición por imagen (ARKit cambia la exposición entre fotos). Solo ganancia: con un
        # término aditivo el modelo "pintaba" las paredes blancas con él en lugar de crear gaussianas.
        expo = torch.nn.Parameter(torch.zeros(len(vistas), 3, device=dev))
        optim_expo = torch.optim.Adam([expo], lr=1e-3)

        # Densificación contenida: con 12 GB de GPU, pasar de ~3 millones de gaussianas desborda la memoria.
        estrategia = DefaultStrategy(refine_start_iter=500, refine_stop_iter=int(a.pasos * 0.4), reset_every=3000,
                                     refine_every=100, grow_grad2d=a.umbral_crecimiento, prune_opa=0.02,
                                     absgrad=False, key_for_gradient="gradient_2dgs", verbose=False)
        estrategia.check_sanity(splats, optim)
        estado = estrategia.initialize_state(scene_scale=escena)

        registro(f"Entrenando {a.pasos} pasos con {n} gaussianas iniciales (escala de escena {escena:.2f} m)…")
        t0 = time.time()
        orden = np.random.default_rng(1).permutation(np.tile(np.arange(len(vistas)), a.pasos // len(vistas) + 1))
        for paso in range(a.pasos):
            iv = int(orden[paso])
            v = vistas[iv]
            gt = v["img"].to(dev).float() / 255
            K = torch.from_numpy(v["K"]).to(dev)[None]
            w2c = torch.from_numpy(v["w2c"]).to(dev)[None]
            sh = min(grado, paso // 1000)
            render, alfa, normales, normales_prof, distorsion, mediana, info = rasterization_2dgs(
                means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]),
                colors=torch.cat([splats["sh0"], splats["shN"]], 1), viewmats=w2c, Ks=K,
                width=v["ancho"], height=v["alto"], sh_degree=sh, render_mode="RGB+ED",
                distloss=paso >= 3000, packed=False, near_plane=0.05, far_plane=20.0)
            estrategia.step_pre_backward(splats, optim, estado, paso, info)
            color = render[0, ..., :3]
            e = expo[iv]
            color_corr = color * torch.exp(e)
            l1 = (color_corr - gt).abs().mean()
            perdida = 0.8 * l1 + 0.2 * (1 - ssim(color_corr.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
            perdida = perdida + 0.01 * (e ** 2).sum()
            # Profundidad LiDAR (relativa) a 1/4 de resolución.
            prof = render[0, ..., 3:4].permute(2, 0, 1)[None]
            prof_baja = F.avg_pool2d(prof, 4)[0, 0]
            ref = v["prof"].to(dev)
            hh, ww = min(ref.shape[0], prof_baja.shape[0]), min(ref.shape[1], prof_baja.shape[1])
            ref, prof_baja = ref[:hh, :ww], prof_baja[:hh, :ww]
            valida = ref > 0
            if valida.any():
                rel = ((prof_baja - ref).abs() / ref)[valida]
                perdida = perdida + a.peso_profundidad * rel.clamp(max=0.5).mean()
                # La habitación es cerrada: donde el LiDAR ve superficie, el render debe ser opaco.
                alfa_baja = F.avg_pool2d(alfa[0].permute(2, 0, 1)[None], 4)[0, 0][:hh, :ww]
                perdida = perdida + 0.1 * (1 - alfa_baja[valida]).mean()
            if paso >= 7000:
                # gsplat devuelve las normales de la superficie sin la dimensión de lote.
                n_sup = normales_prof.reshape(normales[0].shape) * alfa[0].detach()
                perdida = perdida + 0.05 * (1 - (normales[0] * n_sup).sum(-1)).mean()
            if paso >= 3000:
                perdida = perdida + 1e-2 * distorsion.mean()
            perdida.backward()
            # Discos degenerados dan NaN en el backward de 2DGS: se anulan esos gradientes y se acotan las escalas.
            for prm in list(splats.values()) + [expo]:
                if prm.grad is not None:
                    prm.grad.nan_to_num_(0.0, 0.0, 0.0)
            for o in optim.values():
                o.step()
                o.zero_grad(set_to_none=True)
            with torch.no_grad():
                splats["scales"].clamp_(math.log(1e-4), math.log(0.3))
            optim_expo.step()
            optim_expo.zero_grad(set_to_none=True)
            decaimiento.step()
            estrategia.step_post_backward(splats, optim, estado, paso, info, packed=False)
            if paso % 1000 == 0 or paso == a.pasos - 1:
                registro(f"  paso {paso:6d}  pérdida {perdida.item():.4f}  L1 {l1.item():.4f}  "
                         f"gaussianas {len(splats['means'])}  "
                         f"inválidas {torch.isnan(splats['means']).any(-1).sum().item()}  "
                         f"({(time.time() - t0) / 60:.1f} min)")
        torch.save({k: v.detach().cpu() for k, v in splats.items()}, destino / "splats.pt")
        guardar_splats(destino / "splats.ply", splats)
        registro(f"Entrenamiento listo en {(time.time() - t0) / 60:.1f} min")

    extraer_malla(destino, vistas, escena, a.voxel, registro, dev)


@torch.no_grad()
def extraer_malla(destino: Path, vistas, escena: float, voxel: float, registro, dev):
    """Fusión TSDF (en la GPU) de la profundidad mediana y el color renderizados desde todas las vistas, como en
    2DGS, y marching cubes. (La TSDF de Open3D en Windows devuelve mallas vacías, por eso va hecha aquí.)"""
    from gsplat import rasterization_2dgs
    from skimage.measure import marching_cubes

    datos = torch.load(destino / "splats.pt")
    validas = ~torch.cat([torch.isnan(v.reshape(len(v), -1)) for v in datos.values()], 1).any(1)
    splats = {k: v[validas].to(dev) for k, v in datos.items()}
    grado = int(math.isqrt(splats["shN"].shape[1] + 1)) - 1
    # Rejilla que cubre las gaussianas (percentiles para ignorar las sueltas).
    m = splats["means"]
    lo = torch.quantile(m[::10], 0.002, dim=0) - 0.05
    hi = torch.quantile(m[::10], 0.998, dim=0) + 0.05
    dims = ((hi - lo) / voxel).ceil().long()
    registro(f"Rejilla TSDF {dims.tolist()} a {voxel * 100:.1f} cm")
    ejes = [lo[i] + (torch.arange(dims[i], device=dev) + 0.5) * voxel for i in range(3)]
    trunc = 4 * voxel
    tsdf = torch.ones(dims.tolist(), device=dev)
    peso = torch.zeros(dims.tolist(), device=dev)
    color = torch.zeros(dims.tolist() + [3], device=dev)
    t0 = time.time()
    for v in vistas:
        render, alfa, _, _, _, mediana, _ = rasterization_2dgs(
            means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]), colors=torch.cat([splats["sh0"], splats["shN"]], 1),
            viewmats=torch.from_numpy(v["w2c"]).to(dev)[None], Ks=torch.from_numpy(v["K"]).to(dev)[None],
            width=v["ancho"], height=v["alto"], sh_degree=grado, render_mode="RGB+ED", packed=False,
            near_plane=0.05, far_plane=20.0)
        rgb = render[0, ..., :3].clamp(0, 1)
        prof = mediana[0, ..., 0].clone()
        prof[(alfa[0, ..., 0] < 0.5) | (prof > min(6.0, 2 * escena))] = 0
        K = torch.from_numpy(v["K"]).to(dev)
        M = torch.from_numpy(v["w2c"]).to(dev)
        # Por planos en z para no agotar memoria.
        for k in range(int(dims[2])):
            X, Y = torch.meshgrid(ejes[0], ejes[1], indexing="ij")
            pw = torch.stack([X, Y, torch.full_like(X, ejes[2][k].item())], -1).reshape(-1, 3)
            pc = pw @ M[:3, :3].T + M[:3, 3]
            z = pc[:, 2]
            u = (pc[:, 0] / z.clamp(min=1e-6) * K[0, 0] + K[0, 2]).round().long()
            w = (pc[:, 1] / z.clamp(min=1e-6) * K[1, 1] + K[1, 2]).round().long()
            ok = (z > 0.05) & (u >= 0) & (u < v["ancho"]) & (w >= 0) & (w < v["alto"])
            if not ok.any():
                continue
            idx = ok.nonzero()[:, 0]
            d = prof[w[idx], u[idx]]
            sdf = d - z[idx]
            sel = (d > 0) & (sdf > -trunc)
            idx, sdf = idx[sel], (sdf[sel] / trunc).clamp(max=1)
            c_ = rgb[w[idx], u[idx]]
            ix, iy = idx // int(dims[1]), idx % int(dims[1])
            t_, p_ = tsdf[ix, iy, k], peso[ix, iy, k]
            tsdf[ix, iy, k] = (t_ * p_ + sdf) / (p_ + 1)
            color[ix, iy, k] = (color[ix, iy, k] * p_[:, None] + c_) / (p_ + 1)[:, None]
            peso[ix, iy, k] = p_ + 1
    registro(f"TSDF fusionada en {time.time() - t0:.0f} s")
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    vol = tsdf.cpu().numpy()
    observado = peso.cpu().numpy() > 0
    vol[~observado] = 1
    del tsdf, peso
    # La máscara evita la "capa fantasma" entre lo observado y los vóxeles que ninguna cámara vio (detrás de cada
    # superficie), que duplicaba suelo y paredes unos centímetros por detrás.
    vert, caras, _, _ = marching_cubes(vol, level=0.0, spacing=(voxel,) * 3, allow_degenerate=False, mask=observado)
    registro(f"  marching cubes: {len(vert)} vértices, {len(caras)} triángulos")
    ijk = np.clip((vert / voxel).round().astype(int), 0, np.array(vol.shape) - 1)
    col = (color.cpu().numpy()[ijk[:, 0], ijk[:, 1], ijk[:, 2]] * 255).astype(np.uint8)
    vert = (vert + lo.cpu().numpy()).astype(np.float32)
    # Fuera las componentes pequeñas (restos flotantes).
    n = len(vert)
    a = np.concatenate([caras[:, 0], caras[:, 1]])
    b = np.concatenate([caras[:, 1], caras[:, 2]])
    _, etiqueta = connected_components(coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n)), directed=False)
    tam = np.bincount(etiqueta[caras[:, 0]], minlength=etiqueta.max() + 1)
    caras = caras[tam[etiqueta[caras[:, 0]]] >= 500]
    usados = np.unique(caras)
    nuevo = -np.ones(n, np.int64)
    nuevo[usados] = np.arange(len(usados))
    vert, col, caras = vert[usados], col[usados], nuevo[caras].astype(np.int32)  # normales hacia el espacio libre
    guardar_malla_ply(destino / "malla_2dgs.ply", vert, col, caras)
    registro(f"Malla: {len(vert)} vértices, {len(caras)} triángulos "
             f"({time.time() - t0:.0f} s) → {destino / 'malla_2dgs.ply'}")


def guardar_malla_ply(ruta: Path, vert: np.ndarray, col: np.ndarray, caras: np.ndarray):
    vtipo = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.empty(len(vert), vtipo)
    v["x"], v["y"], v["z"] = vert.T
    v["r"], v["g"], v["b"] = col.T
    f = np.empty(len(caras), np.dtype([("n", "u1"), ("i", "<i4", 3)]))
    f["n"], f["i"] = 3, caras
    with open(ruta, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  f"element face {len(f)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        fh.write(v.tobytes())
        fh.write(f.tobytes())


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    main()
