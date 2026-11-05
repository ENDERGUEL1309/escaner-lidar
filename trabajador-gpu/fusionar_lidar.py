r"""Malla LiDAR a color a partir de la grabaciÃ³n de una habitaciÃ³n (modo Â«Modelar habitaciÃ³nÂ»).

    .venv-splat\Scripts\python fusionar_lidar.py "C:\â€¦\Habitacion_â€¦"

La app guarda en fotogramas/ solo las fotos que pasaron los filtros (nÃ­tidas, bien expuestas, sin reflejos fuertes,
giro lento y seguimiento normal), cada una con su LiDAR a resoluciÃ³n completa, su confianza y su pose de ARKit.
AquÃ­ se funden en un volumen TSDF y se escribe malla_color.ply, que es lo que procesar_escaneo.py espera de un
escaneo de entorno: con esto, la habitaciÃ³n pasa por la misma fotogrametrÃ­a y el mismo render fotorrealista.

La fusiÃ³n estÃ¡ hecha con PyTorch (GPU si la hay) y marching cubes de scikit-image: la TSDF de Open3D 0.20 para
Windows devuelve mallas vacÃ­as incluso con su propio ejemplo. Solo se usan lecturas de confianza media o alta y
hasta `--max-prof` metros (mÃ¡s lejos el LiDAR pierde precisiÃ³n).
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from escaner_ia.camaras import arkit_a_opencv, leer_profundidad, matriz_columnas  # noqa: E402

MAX_VOXELES = 80_000_000


def cargar(carpeta: Path, max_prof: float):
    """Fotogramas con profundidad: (mapa hÃ—w en metros, color hÃ—wÃ—3 al tamaÃ±o del mapa Ã—2, K del mapa, mundoâ†’cÃ¡mara)."""
    fotos = json.loads((carpeta / "fotogramas" / "camaras.json").read_text("utf-8"))["fotos"]
    vistas = []
    for foto in fotos:
        info = foto.get("profundidad")
        if not isinstance(info, dict):
            continue
        prof = leer_profundidad(carpeta / "fotogramas" / info["archivo"], info)
        if prof is None:
            continue
        mapa = np.where((prof.mapa > 0.1) & (prof.mapa <= max_prof), prof.mapa, 0).astype(np.float32)
        h, w = mapa.shape
        color = np.asarray(Image.open(carpeta / "fotogramas" / foto["archivo"]).convert("RGB")
                           .resize((w * 2, h * 2), Image.BILINEAR))
        K = np.array([prof.fx, prof.fy, prof.cx, prof.cy], np.float32)
        vistas.append((mapa, color, K, arkit_a_opencv(matriz_columnas(foto["cameraToWorld"])).astype(np.float32)))
    return vistas


def limites(vistas, paso: int = 8) -> tuple[np.ndarray, np.ndarray]:
    """Caja de la escena con los puntos LiDAR (percentiles, para que una lectura perdida no la infle)."""
    puntos = []
    for mapa, _, K, w2c in vistas:
        v, u = np.mgrid[0:mapa.shape[0]:paso, 0:mapa.shape[1]:paso]
        z = mapa[v, u]
        ok = z > 0
        x = (u[ok] - K[2]) / K[0] * z[ok]
        y = (v[ok] - K[3]) / K[1] * z[ok]
        cam = np.stack([x, y, z[ok]], 1)
        c2w = np.linalg.inv(w2c)
        puntos.append(cam @ c2w[:3, :3].T + c2w[:3, 3])
    p = np.concatenate(puntos)
    return np.percentile(p, 0.5, 0) - 0.1, np.percentile(p, 99.5, 0) + 0.1


def fusionar(carpeta: Path, voxel: float = 0.01, max_prof: float = 4.0, salida: Path | None = None) -> tuple[int, int]:
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components
    from skimage.measure import marching_cubes

    from splat2d_local import guardar_malla_ply

    vistas = cargar(carpeta, max_prof)
    if not vistas:
        raise SystemExit("No hay fotogramas con profundidad en fotogramas/camaras.json")
    lo, hi = limites(vistas)
    while np.prod(np.ceil((hi - lo) / voxel)) > MAX_VOXELES:
        voxel *= 1.25
    dims = np.ceil((hi - lo) / voxel).astype(int)
    trunc = 4 * voxel
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"FusiÃ³n LiDAR: {len(vistas)} fotogramas, volumen {dims.tolist()} con vÃ³xel de {voxel * 100:.1f} cm "
          f"en {dev}", flush=True)

    n = int(np.prod(dims))
    tsdf = torch.ones(n, device=dev)
    peso = torch.zeros(n, device=dev)
    color = torch.zeros(n, 3, device=dev)
    # Centros de los vÃ³xeles por bloques de planos z, para no tener todos a la vez en memoria.
    gx, gy = torch.meshgrid(torch.arange(dims[0], device=dev), torch.arange(dims[1], device=dev), indexing="ij")
    # Índices enteros: en float32 un índice lineal por encima de 2^24 (16,7 millones de vóxeles) se redondea y el
    # LiDAR de muchas fotos acababa escrito en vóxeles equivocados, a metros de su sitio.
    plano = torch.stack([gx.flatten(), gy.flatten()], 1).long()
    lo_t = torch.tensor(lo, device=dev, dtype=torch.float32)
    t0 = time.time()

    for k, (mapa, imagen, K, w2c) in enumerate(vistas):
        prof = torch.from_numpy(mapa).to(dev)
        img = torch.from_numpy(np.array(imagen)).to(dev).float()
        R = torch.from_numpy(w2c[:3, :3]).to(dev)
        t = torch.from_numpy(w2c[:3, 3]).to(dev)
        h, w = mapa.shape
        bloque = max(1, 4_000_000 // len(plano))
        for z0 in range(0, dims[2], bloque):
            zs = range(z0, min(int(dims[2]), z0 + bloque))
            idx = torch.cat([torch.cat([plano, torch.full((len(plano), 1), z, device=dev, dtype=torch.long)], 1) for z in zs])
            mundo = lo_t + (idx.float() + 0.5) * voxel
            cam = mundo @ R.T + t
            z = cam[:, 2]
            delante = z > 0.1
            u = torch.round(cam[:, 0] / z.clamp(min=1e-3) * K[0] + K[2]).long()
            v = torch.round(cam[:, 1] / z.clamp(min=1e-3) * K[1] + K[3]).long()
            dentro = delante & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            d = torch.zeros_like(z)
            d[dentro] = prof[v[dentro], u[dentro]]
            sdf = d - z
            util = dentro & (d > 0) & (sdf > -trunc)
            if not util.any():
                continue
            lineal = (idx[:, 0] * int(dims[1] * dims[2]) + idx[:, 1] * int(dims[2]) + idx[:, 2])[util]
            valor = (sdf[util] / trunc).clamp(max=1)
            # Las lecturas lejanas y de canto pesan menos: el error del LiDAR crece con la distancia.
            p = 1 / d[util].clamp(min=0.5)
            col = img[(v[util] * 2).clamp(max=2 * h - 1), (u[util] * 2).clamp(max=2 * w - 1)]
            nuevo = peso[lineal] + p
            tsdf[lineal] = (tsdf[lineal] * peso[lineal] + valor * p) / nuevo
            color[lineal] = (color[lineal] * peso[lineal, None] + col * p[:, None]) / nuevo[:, None]
            peso[lineal] = nuevo
        if (k + 1) % 50 == 0:
            print(f"FusiÃ³n LiDAR: {k + 1}/{len(vistas)} ({time.time() - t0:.0f} s)", flush=True)

    volumen = tsdf.reshape(*dims).cpu().numpy()
    visto = (peso.reshape(*dims) > 0).cpu().numpy()
    vert, caras, _, _ = marching_cubes(volumen, level=0.0, mask=visto, allow_degenerate=False)
    # El vÃ©rtice toma el color del vÃ³xel mÃ¡s cercano.
    vi = np.clip(np.round(vert).astype(int), 0, dims - 1)
    colores = color.reshape(*dims, 3).cpu().numpy()[vi[:, 0], vi[:, 1], vi[:, 2]]
    vert = lo + (vert + 0.5) * voxel

    # Restos sueltos (polvo de lecturas malas): fuera los grupos de menos de 200 triÃ¡ngulos.
    aristas = np.concatenate([caras[:, [0, 1]], caras[:, [1, 2]]])
    grafo = coo_matrix((np.ones(len(aristas)), (aristas[:, 0], aristas[:, 1])), shape=(len(vert), len(vert)))
    _, etiqueta = connected_components(grafo, directed=False)
    tamano = np.bincount(etiqueta[caras[:, 0]], minlength=etiqueta.max() + 1)
    caras = caras[tamano[etiqueta[caras[:, 0]]] >= 200]
    usados = np.unique(caras)
    nuevo_indice = np.full(len(vert), -1)
    nuevo_indice[usados] = np.arange(len(usados))

    guardar_malla_ply(salida or carpeta / "malla_color.ply", vert[usados].astype(np.float32),
                      np.clip(colores[usados], 0, 255).astype(np.uint8), nuevo_indice[caras].astype(np.int32))
    return len(vistas), len(caras)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    p.add_argument("--voxel", type=float, default=0.01, help="tamaÃ±o del vÃ³xel en metros")
    p.add_argument("--max-prof", type=float, default=4.0)
    p.add_argument("--salida", help="otra ruta para la malla (por defecto <carpeta>/malla_color.ply)")
    a = p.parse_args()
    usadas, triangulos = fusionar(Path(a.carpeta).resolve(), a.voxel, a.max_prof,
                                  salida=Path(a.salida) if a.salida else None)
    print(f"FusiÃ³n LiDAR lista: {usadas} fotogramas â†’ malla_color.ply con {triangulos} triÃ¡ngulos", flush=True)


if __name__ == "__main__":
    main()
