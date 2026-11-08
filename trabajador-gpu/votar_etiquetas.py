r"""Etiqueta semántica de cada punto 3D por votación entre vistas (después de segmentar_fotos.py).

    .venv-splat\Scripts\python votar_etiquetas.py "C:\…\Habitacion_…"

Cada punto se proyecta en todas las fotos. Solo vota una foto que lo ve de verdad (su LiDAR mide esa distancia; si
mide menos, hay algo delante) y vota con la confianza de SAM3, más fuerte cuanto más cerca está la cámara. Gana la
clase con más peso. Así una foto en la que SAM3 se confundió queda en minoría frente a las que acertaron.

Se etiquetan los vértices de la malla LiDAR (malla_color.ply) y, si existe, cada splat del render 3DGS
(splats_render/escena.splat).

Resultado en <escaneo>/segmentacion/: etiquetas_lidar.npz (etiqueta y confianza por vértice), malla_etiquetada.ply
(la malla LiDAR coloreada por clase) y escena_etiquetada.splat (los splats coloreados por clase, para abrir en
SuperSplat o en la app y compararlos con los reales).
"""

from __future__ import annotations

import argparse
import colorsys
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).parent))
from remodelar_envolvente import cargar_fotos, leer_malla  # noqa: E402


@torch.no_grad()
def votar(puntos: np.ndarray, fotos, n_clases: int, dev, bloque: int = 400_000):
    """Etiqueta (0 = sin datos) y confianza (peso de la ganadora / peso total) de cada punto."""
    etiqueta = np.zeros(len(puntos), np.uint8)
    confianza = np.zeros(len(puntos), np.float32)
    for i in range(0, len(puntos), bloque):
        P = torch.from_numpy(puntos[i:i + bloque]).float().to(dev)
        votos = torch.zeros(len(P), n_clases + 1, device=dev)
        for f in fotos:
            if "etiquetas" not in f:
                continue
            R, t = f["w2c"][:3, :3], f["w2c"][:3, 3]
            cam = P @ R.T + t
            z = cam[:, 2]
            fx, fy, cx, cy = f["K"]
            h, w = f["etiquetas"].shape
            u = (cam[:, 0] / z.clamp(min=1e-3) * fx + cx).long()
            v = (cam[:, 1] / z.clamp(min=1e-3) * fy + cy).long()
            ok = (z > 0.1) & (u >= 0) & (u < w) & (v >= 0) & (v < h)
            pfx, pfy, pcx, pcy = f["Kp"]
            ph, pw = f["prof"].shape
            pu = (cam[:, 0] / z.clamp(min=1e-3) * pfx + pcx).long().clamp(0, pw - 1)
            pv = (cam[:, 1] / z.clamp(min=1e-3) * pfy + pcy).long().clamp(0, ph - 1)
            d = f["prof"][pv, pu]
            ok &= (d > 0) & ((d - z).abs() < 0.05 + 0.02 * z)
            idx = ok.nonzero()[:, 0]
            if not len(idx):
                continue
            et = f["etiquetas"][v[idx], u[idx]].long()
            peso = f["confianza"][v[idx], u[idx]] / z[idx].clamp(min=0.5)
            votos.index_put_((idx, et), peso, accumulate=True)
        votos[:, 0] = 0                                  # «sin etiqueta» no vota
        total = votos.sum(1)
        mejor, cual = votos.max(1)
        sin = total < 0.3
        cual[sin] = 0
        etiqueta[i:i + bloque] = cual.byte().cpu().numpy()
        confianza[i:i + bloque] = torch.where(sin, torch.zeros_like(total), mejor / total.clamp(min=1e-6)).cpu().numpy()
    return etiqueta, confianza


def paleta(n: int) -> np.ndarray:
    """Un color vivo y distinto por clase (0 = gris)."""
    colores = [(90, 90, 90)]
    for k in range(n):
        r, g, b = colorsys.hsv_to_rgb((k * 0.618034) % 1.0, 0.75, 0.95)
        colores.append((int(r * 255), int(g * 255), int(b * 255)))
    return np.array(colores, np.uint8)


def guardar_malla(ruta: Path, v: np.ndarray, col: np.ndarray, caras: np.ndarray):
    vt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    vv = np.empty(len(v), vt)
    vv["x"], vv["y"], vv["z"] = v.T
    vv["r"], vv["g"], vv["b"] = col.T
    ff = np.empty(len(caras), np.dtype([("n", "u1"), ("i", "<i4", 3)]))
    ff["n"], ff["i"] = 3, caras
    with open(ruta, "wb") as fh:
        fh.write((f"ply\nformat binary_little_endian 1.0\nelement vertex {len(v)}\n"
                  "property float x\nproperty float y\nproperty float z\n"
                  "property uchar red\nproperty uchar green\nproperty uchar blue\n"
                  f"element face {len(caras)}\nproperty list uchar int vertex_indices\nend_header\n").encode())
        fh.write(vv.tobytes())
        fh.write(ff.tobytes())


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    a = p.parse_args()
    origen = Path(a.carpeta).resolve()
    destino = origen / "segmentacion"
    clases = json.loads((destino / "clases.json").read_text("utf-8"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    t0 = time.time()
    fotos = [f for f in cargar_fotos(origen, dev) if "etiquetas" in f]
    print(f"{len(fotos)} fotos segmentadas con LiDAR", flush=True)
    colores = paleta(len(clases))

    v, caras = leer_malla(origen / "malla_color.ply")
    et, conf = votar(v, fotos, len(clases), dev)
    np.savez_compressed(destino / "etiquetas_lidar.npz", etiqueta=et, confianza=conf)
    guardar_malla(destino / "malla_etiquetada.ply", v.astype(np.float32), colores[et], caras.astype(np.int32))
    cuenta = np.bincount(et, minlength=len(clases) + 1)
    resumen = ", ".join(f"{c['nombre']} {cuenta[c['etiqueta']] / len(et):.0%}" for c in clases
                        if cuenta[c["etiqueta"]] / len(et) >= 0.005)
    print(f"Malla LiDAR: {len(et)} puntos, sin etiqueta {cuenta[0] / len(et):.0%}; {resumen}", flush=True)

    # Splats del render 3DGS (formato .splat: 32 bytes; posición 3f, escala 3f, color RGBA, giro 4 bytes).
    splat = origen / "splats_render" / "escena.splat"
    if splat.exists():
        datos = np.frombuffer(splat.read_bytes(), np.uint8).reshape(-1, 32).copy()
        pos = datos[:, :12].copy().view(np.float32).reshape(-1, 3)
        es, _ = votar(pos.astype(np.float64), fotos, len(clases), dev)
        datos[:, 24:27] = colores[es]
        (destino / "escena_etiquetada.splat").write_bytes(datos.tobytes())
        print(f"Splats: {len(es)} etiquetados, sin etiqueta {np.mean(es == 0):.0%}", flush=True)
    leyenda = [{"etiqueta": 0, "nombre": "sin etiqueta", "color": colores[0].tolist()}] + \
        [{"etiqueta": c["etiqueta"], "nombre": c["nombre"], "color": colores[c["etiqueta"]].tolist()} for c in clases]
    (destino / "leyenda.json").write_text(json.dumps(leyenda, indent=1, ensure_ascii=False), "utf-8")
    print(f"Votación lista en {time.time() - t0:.0f} s → {destino}", flush=True)


if __name__ == "__main__":
    main()
