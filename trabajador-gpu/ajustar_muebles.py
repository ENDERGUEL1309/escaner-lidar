r"""Remodelado de la habitación, paso 4: encajar cada mueble generado en su sitio, comprobarlo y pintarlo.

    .venv-splat\Scripts\python ajustar_muebles.py "C:\…\Habitacion_…"

Para cada mueble con forma generada (mueble_N_ia.glb):
1. Se simplifica, se escala a las medidas reales de su caja (RoomPlan o separación automática) y se prueba con los
   cuatro giros posibles: queda el que mejor coincide con los puntos LiDAR del mueble.
2. Comprobación: distancia de cada punto LiDAR a la superficie del modelo. Verificado si la mediana es ≤ 4 cm y el
   90 % está a ≤ 12 cm; si no, queda marcado como dudoso (se ve, pero conviene revisarlo o escanearlo en detalle).
3. Color: cada vértice toma el color de las fotos que lo ven de frente y sin nada delante (prueba con el LiDAR).
Los muebles sin forma generada quedan como una caja limpia con su color medio. Los escaneados en detalle con Object
Capture los coloca directamente el montaje de Blender.

Resultado: remodelado/objetos/mueble_N.ply (mundo de ARKit, color por vértice) y ajuste.json.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).parent))
from remodelar_envolvente import cargar_fotos, ganancias  # noqa: E402
from splat2d_local import guardar_malla_ply  # noqa: E402


def giro_y(a: float) -> np.ndarray:
    c, s = np.cos(a), np.sin(a)
    return np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])


def cargar_ia(ruta: Path, caras_max: int = 80_000):
    import fast_simplification
    import trimesh

    escena = trimesh.load(ruta, force="mesh")
    v, f = np.asarray(escena.vertices, np.float64), np.asarray(escena.faces, np.int32)
    if len(f) > caras_max:
        v, f = fast_simplification.simplify(v.astype(np.float32), f, target_reduction=1 - caras_max / len(f))
    return np.asarray(v, np.float64), np.asarray(f, np.int64)


def muestras(v, f, n=40_000, semilla=0):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    area = np.linalg.norm(np.cross(b - a, c - a), axis=1)
    rng = np.random.default_rng(semilla)
    i = rng.choice(len(f), n, p=area / area.sum())
    r1, r2 = rng.random((2, n))
    s = np.sqrt(r1)
    return (1 - s)[:, None] * a[i] + (s * (1 - r2))[:, None] * b[i] + (s * r2)[:, None] * c[i]


def encajar(v, f, m, lidar):
    """Mejor de los cuatro giros: forma escalada a la caja y colocada; devuelve vértices en el mundo y métricas."""
    tam, pos, yaw = np.array(m["tam"]), np.array(m["posicion"]), m["yaw"]
    centrado = v - (v.min(0) + v.max(0)) / 2
    mejor = None
    arbol_lidar = cKDTree(lidar) if len(lidar) else None
    for k in range(4):
        r = centrado @ giro_y(k * np.pi / 2).T
        ext = r.max(0) - r.min(0)
        r = r / np.maximum(ext, 1e-6) * tam                  # mismas medidas que la caja real
        r[:, 1] -= r[:, 1].min()                              # apoyado en el suelo
        mundo = r @ giro_y(yaw).T + pos
        if arbol_lidar is None:
            return mundo, None
        sup = muestras(mundo, f, 30_000, k)
        d_lidar = cKDTree(sup).query(lidar)[0]               # cada punto medido → superficie del modelo
        d_modelo = arbol_lidar.query(sup)[0]                  # superficie del modelo → lo medido (cubre la parte vista)
        nota = np.median(d_lidar) + 0.5 * np.percentile(d_modelo, 50)
        if mejor is None or nota < mejor[0]:
            mejor = (nota, mundo, {"giro": k * 90, "mediana_cm": round(float(np.median(d_lidar)) * 100, 1),
                                   "p90_cm": round(float(np.percentile(d_lidar, 90)) * 100, 1)})
    return mejor[1], mejor[2]


def normales_vertice(v, f):
    n = np.zeros_like(v)
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    for i in range(3):
        np.add.at(n, f[:, i], fn)
    return n / np.maximum(np.linalg.norm(n, axis=1, keepdims=True), 1e-9)


@torch.no_grad()
def colorear(v, f, fotos, gan, dev, top=3):
    """Color por vértice desde las fotos: de frente, cerca y sin nada delante (según el LiDAR de cada foto)."""
    P = torch.from_numpy(v).float().to(dev)
    N = torch.from_numpy(normales_vertice(v, f)).float().to(dev)
    mejores_p = torch.zeros(len(P), top, device=dev)
    mejores_c = torch.zeros(len(P), top, 3, device=dev)
    for foto, g in zip(fotos, gan):
        R, t = foto["w2c"][:3, :3], foto["w2c"][:3, 3]
        cam = P @ R.T + t
        z = cam[:, 2]
        fx, fy, cx, cy = foto["K"]
        h, w = foto["img"].shape[:2]
        u, vv = cam[:, 0] / z.clamp(min=1e-3) * fx + cx, cam[:, 1] / z.clamp(min=1e-3) * fy + cy
        ok = (z > 0.2) & (u >= 0) & (u < w - 1) & (vv >= 0) & (vv < h - 1)
        pfx, pfy, pcx, pcy = foto["Kp"]
        ph, pw = foto["prof"].shape
        pu = (cam[:, 0] / z.clamp(min=1e-3) * pfx + pcx).long().clamp(0, pw - 1)
        pv = (cam[:, 1] / z.clamp(min=1e-3) * pfy + pcy).long().clamp(0, ph - 1)
        d = foto["prof"][pv, pu]
        ok &= (d > 0) & ((d - z).abs() < 0.06)
        centro = -R.T @ t
        frente = (F.normalize(centro - P, dim=1) * N).sum(1).abs()   # abs: las normales generadas pueden ir al revés
        p = torch.where(ok & (frente > 0.2), frente ** 2 / z.clamp(min=0.3) ** 2, torch.zeros_like(z))
        if not (p > 0).any():
            continue
        col = F.grid_sample(foto["img"].permute(2, 0, 1)[None],
                            torch.stack([u / (w - 1) * 2 - 1, vv / (h - 1) * 2 - 1], 1)[None, None],
                            mode="bilinear", align_corners=True)[0, :, 0].T * g
        todas_p = torch.cat([mejores_p, p[:, None]], 1)
        todas_c = torch.cat([mejores_c, col[:, None]], 1)
        orden = todas_p.argsort(1, descending=True)[:, :top]
        mejores_p = todas_p.gather(1, orden)
        mejores_c = todas_c.gather(1, orden[..., None].expand(-1, -1, 3))
    peso = mejores_p ** 2
    total = peso.sum(1)
    color = ((mejores_c * peso[..., None]).sum(1) / total.clamp(min=1e-12)[:, None]).cpu().numpy()
    visto = (total > 0).cpu().numpy()
    if visto.any() and not visto.all():
        # Lo que ninguna foto ve (la parte de atrás) toma el color del vértice visto más cercano.
        cerca = cKDTree(v[visto]).query(v[~visto])[1]
        color[~visto] = color[visto][cerca]
    elif not visto.any():
        color[:] = 0.6
    return (np.clip(color, 0, 1) * 255).astype(np.uint8), float(visto.mean())


def caja_malla(m):
    tam, pos, yaw = np.array(m["tam"]), np.array(m["posicion"]), m["yaw"]
    s = np.array([[x, y, z] for x in (-.5, .5) for y in (0, 1) for z in (-.5, .5)]) * tam
    f = np.array([[0, 2, 3], [0, 3, 1], [4, 5, 7], [4, 7, 6], [0, 1, 5], [0, 5, 4],
                  [2, 6, 7], [2, 7, 3], [0, 4, 6], [0, 6, 2], [1, 3, 7], [1, 7, 5]])
    return s @ giro_y(yaw).T + pos, f


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    a = p.parse_args()
    origen = Path(a.carpeta).resolve()
    carpeta = origen / "remodelado" / "objetos"
    info = json.loads((carpeta / "objetos.json").read_text("utf-8"))
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    fotos = cargar_fotos(origen, dev)
    gan = ganancias(fotos)
    informe = []
    for m in info["muebles"]:
        if m["detalle"]:
            informe.append({"clave": m["clave"], "nombre": m["nombre"], "estado": "escaneo en detalle"})
            continue
        lidar = np.load(carpeta / f"{m['clave']}_puntos.npy").astype(np.float64)
        ia = carpeta / f"{m['clave']}_ia.glb"
        if ia.exists():
            v, f = cargar_ia(ia)
            v, metricas = encajar(v, f, m, lidar)
            estado = ("verificado" if metricas and metricas["mediana_cm"] <= 4 and metricas["p90_cm"] <= 12
                      else "dudoso")
        else:
            (v, f), metricas, estado = caja_malla(m), None, "caja (sin vista completa)"
        color, visto = colorear(v, f, fotos, gan, dev)
        guardar_malla_ply(carpeta / f"{m['clave']}.ply", v.astype(np.float32), color, f.astype(np.int32))
        linea = {"clave": m["clave"], "nombre": m["nombre"], "tipo": m["tipo"], "estado": estado,
                 "color_visto": round(visto, 3), **(metricas or {})}
        informe.append(linea)
        detalle = (f"LiDAR a {metricas['mediana_cm']} cm de mediana, 90 % a ≤ {metricas['p90_cm']} cm, giro "
                   f"{metricas['giro']}°" if metricas else "sin forma generada")
        print(f"  {m['nombre']}: {estado} — {detalle}; color visto en el {visto * 100:.0f} %", flush=True)
    (carpeta / "ajuste.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()
