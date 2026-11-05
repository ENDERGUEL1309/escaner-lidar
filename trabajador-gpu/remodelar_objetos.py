r"""Remodelado de la habitación, paso 2: separar los muebles y preparar para cada uno su mejor foto recortada.

    .venv-splat\Scripts\python remodelar_objetos.py "C:\…\Habitacion_…"

Necesita remodelado/envolvente.json (remodelar_envolvente.py). De la malla LiDAR se quita lo que está pegado a
paredes, suelo y techo; lo que queda dentro de la habitación se agrupa por contacto (vóxeles de 3 cm) y cada grupo
grande es un objeto. Para cada objeto se elige la foto que lo ve más entero, más cerca y más de frente, y se recorta
su silueta exacta proyectando sus puntos LiDAR (con la prueba de oclusión del LiDAR de esa foto): así el generador
3D recibe el mueble aislado, sin el fondo ni lo que tenga delante.

Resultado en <escaneo>/remodelado/objetos/: objeto_N.png (RGBA, fondo transparente) y objetos.json con la caja de
cada objeto en el sistema de la habitación (eje largo, eje ancho, vertical) y la foto elegida.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from scipy import ndimage

sys.path.insert(0, str(Path(__file__).parent))
from remodelar_envolvente import cargar_fotos, leer_malla  # noqa: E402

VOXEL = 0.03


def ejes_habitacion(env: dict) -> np.ndarray:
    """Base de la habitación (filas): eje del largo, vertical y eje del ancho, en el mundo de ARKit."""
    suelo = env["planos"]["suelo"]
    u, v = np.array(suelo["u"]), np.array(suelo["v"])
    return np.stack([u, np.array([0.0, 1.0, 0.0]), v])


def separar(origen: Path, env: dict, registro=print):
    v, _ = leer_malla(origen / "malla_color.ply")
    B = ejes_habitacion(env)
    c0 = np.array(env["planos"]["suelo"]["centro"])
    local = (v - c0) @ B.T                         # x: largo, y: altura sobre el suelo, z: ancho
    L, A, H = env["largo_m"], env["ancho_m"], env["alto_m"]
    margen = 0.06
    dentro = ((np.abs(local[:, 0]) < L / 2 - margen) & (np.abs(local[:, 2]) < A / 2 - margen)
              & (local[:, 1] > 0.03) & (local[:, 1] < H - margen))
    p = local[dentro]
    registro(f"{dentro.sum()} de {len(v)} puntos LiDAR dentro de la habitación y fuera de sus superficies")
    # Ocupación en vóxeles de 3 cm; los grupos conectados (también en diagonal) son objetos.
    lo = p.min(0)
    idx = np.floor((p - lo) / VOXEL).astype(int)
    rejilla = np.zeros(idx.max(0) + 1, bool)
    rejilla[tuple(idx.T)] = True
    etiquetas, n = ndimage.label(ndimage.binary_closing(rejilla, iterations=1), structure=np.ones((3, 3, 3)))
    grupo = etiquetas[tuple(idx.T)]
    objetos = []
    for g in range(1, n + 1):
        q = p[grupo == g]
        if len(q) < 1500:
            continue
        mn, mx = np.percentile(q, 1, 0), np.percentile(q, 99, 0)
        tam = mx - mn
        if tam.max() < 0.15 or np.prod(tam) < 0.003:
            continue
        objetos.append({"puntos_local": q, "min": mn, "max": mx})
    objetos.sort(key=lambda o: -np.prod(o["max"] - o["min"]))
    return objetos, B, c0


@torch.no_grad()
def mejor_recorte(puntos_mundo: np.ndarray, fotos, destino: Path, nombre: str, dev):
    mundo = torch.from_numpy(puntos_mundo).float().to(dev)
    mejor = None
    for k, f in enumerate(fotos):
        R, t = f["w2c"][:3, :3], f["w2c"][:3, 3]
        cam = mundo @ R.T + t
        z = cam[:, 2]
        fx, fy, cx, cy = f["K"]
        h, w = f["img"].shape[:2]
        u, vv = cam[:, 0] / z.clamp(min=1e-3) * fx + cx, cam[:, 1] / z.clamp(min=1e-3) * fy + cy
        en_foto = (z > 0.2) & (u >= 0) & (u < w) & (vv >= 0) & (vv < h)
        # Visible: el LiDAR de la foto mide (más o menos) ese punto, no algo delante.
        pfx, pfy, pcx, pcy = f["Kp"]
        ph, pw = f["prof"].shape
        pu = (cam[:, 0] / z.clamp(min=1e-3) * pfx + pcx).long().clamp(0, pw - 1)
        pv = (cam[:, 1] / z.clamp(min=1e-3) * pfy + pcy).long().clamp(0, ph - 1)
        d = f["prof"][pv, pu]
        visible = en_foto & (d > 0) & (d > z - 0.06)
        frac = float(visible.float().mean())
        if frac < 0.5:
            continue
        # Entero (sin cortar por el borde), cerca (más píxeles) y visible.
        uu, vvv = u[visible], vv[visible]
        caja = (float(uu.min()), float(vvv.min()), float(uu.max()), float(vvv.max()))
        cortado = caja[0] < 8 or caja[1] < 8 or caja[2] > w - 8 or caja[3] > h - 8
        area = (caja[2] - caja[0]) * (caja[3] - caja[1]) / (w * h)
        nota = frac * min(area, 0.5) * (0.4 if cortado else 1.0)
        if mejor is None or nota > mejor[0]:
            mejor = (nota, k, uu, vvv, caja, frac)
    if mejor is None:
        return None
    _, k, uu, vvv, caja, frac = mejor
    f = fotos[k]
    h, w = f["img"].shape[:2]
    # Máscara: puntos proyectados, engordados y cerrados (la malla LiDAR es más gruesa que un píxel).
    mascara = torch.zeros(h, w, device=dev)
    mascara[vvv.long().clamp(0, h - 1), uu.long().clamp(0, w - 1)] = 1
    radio = max(3, int(0.012 * max(caja[2] - caja[0], caja[3] - caja[1])))
    mascara = F.max_pool2d(mascara[None, None], 2 * radio + 1, 1, radio)
    mascara = -F.max_pool2d(-mascara, 2 * radio + 1, 1, radio)[0, 0]
    m = mascara.cpu().numpy() > 0.5
    m = ndimage.binary_fill_holes(m)
    x0, y0, x1, y1 = (int(v) for v in caja)
    pad = int(0.08 * max(x1 - x0, y1 - y0))
    x0, y0, x1, y1 = max(0, x0 - pad), max(0, y0 - pad), min(w, x1 + pad), min(h, y1 + pad)
    rgb = (f["img"].cpu().numpy() * 255).astype(np.uint8)[y0:y1, x0:x1]
    alfa = (m[y0:y1, x0:x1] * 255).astype(np.uint8)
    lado = max(rgb.shape[:2])
    lienzo = np.zeros((lado, lado, 4), np.uint8)
    oy, ox = (lado - rgb.shape[0]) // 2, (lado - rgb.shape[1]) // 2
    lienzo[oy:oy + rgb.shape[0], ox:ox + rgb.shape[1], :3] = rgb
    lienzo[oy:oy + rgb.shape[0], ox:ox + rgb.shape[1], 3] = alfa
    Image.fromarray(lienzo).resize((768, 768), Image.LANCZOS).save(destino / f"{nombre}.png")
    # Dirección desde el objeto hacia la cámara (en el sistema de la habitación): el «frente» de la imagen.
    centro_cam = (-f["w2c"][:3, :3].T @ f["w2c"][:3, 3]).cpu().numpy()
    return {"foto": k, "visible": round(frac, 3), "camara_mundo": centro_cam.tolist()}


TIPOS = {"table": "mesa", "desk": "escritorio", "bed": "cama", "sofa": "sofá", "chair": "silla",
         "storage": "armario", "appliance": "electrodoméstico"}


def muebles_roomplan(origen: Path, v_mundo: np.ndarray) -> list[dict] | None:
    """Muebles de habitacion.json (RoomPlan): nombre, tipo y caja medida, en el mismo mundo de ARKit que las fotos.
    Los puntos LiDAR de cada uno son los que caen en su caja (con 5 cm de margen)."""
    ruta = origen / "habitacion.json"
    if not ruta.exists():
        return None
    sala = json.loads(ruta.read_text("utf-8"))
    muebles = []
    for m in sala.get("furniture", []):
        pos, yaw, tam = np.array(m["position"], float), float(m["yaw"]), np.array(m["size"], float)
        c, s = np.cos(yaw), np.sin(yaw)
        R = np.array([[c, 0, s], [0, 1, 0], [-s, 0, c]])          # mueble → mundo (giro sobre Y)
        local = (v_mundo - pos) @ R                                 # mundo → mueble
        dentro = ((np.abs(local[:, 0]) < tam[0] / 2 + 0.05) & (np.abs(local[:, 2]) < tam[2] / 2 + 0.05)
                  & (local[:, 1] > 0.02) & (local[:, 1] < tam[1] + 0.05))
        muebles.append({"nombre": m["name"], "tipo": TIPOS.get(m["kind"], m["kind"]), "id": m["id"],
                        "posicion": pos, "yaw": yaw, "tam": tam, "puntos_mundo": v_mundo[dentro],
                        "detalle": (m.get("detail") or {}).get("file"),
                        "matriz_detalle": (m.get("detail") or {}).get("matrix")})
    return muebles


def muebles_separados(origen: Path, env: dict) -> list[dict]:
    """Sin RoomPlan: los grupos de puntos separados por separar(), con su caja alineada a la habitación."""
    objetos, B, c0 = separar(origen, env)
    u = B[0]
    yaw = float(np.arctan2(-u[2], u[0]))                           # el eje x del mueble = eje del largo
    muebles = []
    for i, o in enumerate(objetos):
        tam = o["max"] - o["min"]
        centro = (o["max"] + o["min"]) / 2
        base = np.array([centro[0], 0.0, centro[2]]) @ B + c0      # centro de la planta, a la altura del suelo
        muebles.append({"nombre": f"Objeto {i + 1}", "tipo": "objeto", "id": f"objeto_{i + 1}",
                        "posicion": base, "yaw": yaw, "tam": tam, "puntos_mundo": o["puntos_local"] @ B + c0,
                        "detalle": None, "matriz_detalle": None})
    return muebles


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    p.add_argument("--max-objetos", type=int, default=12)
    a = p.parse_args()
    origen = Path(a.carpeta).resolve()
    env = json.loads((origen / "remodelado" / "envolvente.json").read_text("utf-8"))
    destino = origen / "remodelado" / "objetos"
    destino.mkdir(parents=True, exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    v, _ = leer_malla(origen / "malla_color.ply")
    muebles = muebles_roomplan(origen, v)
    fuente = "RoomPlan (habitacion.json)"
    if muebles is None:
        muebles, fuente = muebles_separados(origen, env), "separación automática de la malla LiDAR"
    print(f"{len(muebles)} muebles según {fuente}", flush=True)
    fotos = cargar_fotos(origen, dev)
    salida = []
    for i, m in enumerate(muebles[:a.max_objetos]):
        clave = f"mueble_{i + 1}"
        recorte = None
        if len(m["puntos_mundo"]) >= 300 and not m["detalle"]:
            recorte = mejor_recorte(m["puntos_mundo"], fotos, destino, clave, dev)
        np.save(destino / f"{clave}_puntos.npy", m["puntos_mundo"].astype(np.float32))
        tam = m["tam"]
        salida.append({"clave": clave, "nombre": m["nombre"], "tipo": m["tipo"], "id": m["id"],
                       "posicion": [round(float(x), 4) for x in m["posicion"]], "yaw": round(m["yaw"], 5),
                       "tam": [round(float(x), 4) for x in tam], "puntos": len(m["puntos_mundo"]),
                       "recorte": recorte, "detalle": m["detalle"], "matriz_detalle": m["matriz_detalle"]})
        origen_modelo = "escaneo en detalle" if m["detalle"] else f"foto {recorte['foto']}" if recorte else "sin vista completa"
        print(f"  {m['nombre']} ({m['tipo']}): {tam[0]:.2f} × {tam[1]:.2f} × {tam[2]:.2f} m, "
              f"{len(m['puntos_mundo'])} puntos LiDAR, {origen_modelo}", flush=True)
    (destino / "objetos.json").write_text(json.dumps({
        "fuente": fuente,
        "sistema": "mundo de ARKit (Y arriba); posicion = centro de la planta en el suelo; yaw sobre Y; "
                   "tam = ancho (x), alto (y), fondo (z) del mueble",
        "muebles": salida}, indent=1, ensure_ascii=False), "utf-8")


if __name__ == "__main__":
    main()
