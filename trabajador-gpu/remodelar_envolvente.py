r"""Remodelado de la habitación, paso 1: la envolvente (suelo, techo y paredes) como planos exactos con textura.

    .venv-splat\Scripts\python remodelar_envolvente.py "C:\…\Habitacion_…"

El escaneo no es el resultado, es la medida: aquí se buscan en la malla LiDAR los planos dominantes (suelo y techo
horizontales; paredes verticales y a escuadra entre sí, como en casi cualquier habitación) y se construye una caja
limpia con ellos. Después cada plano se pinta con las fotos: para cada punto del plano se eligen las fotos que lo
ven de frente y de cerca, y solo si su LiDAR confirma que ese punto es visible (si delante hay un mueble, la foto
no cuenta). Lo que ninguna foto ve limpio se rellena con los alrededores. El resultado no tiene ruido de escaneo ni
muebles pegados a las paredes: los muebles se añaden después como objetos propios.

Resultado en <escaneo>/remodelado/: envolvente.obj + .mtl + una textura PNG por plano, y envolvente.json con las
medidas (largo, ancho y alto de la habitación y la ecuación de cada plano).
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from escaner_ia.camaras import arkit_a_opencv, leer_profundidad, matriz_columnas  # noqa: E402


# ----------------------------------------------------------------------------------------------- malla

def leer_malla(ruta: Path) -> tuple[np.ndarray, np.ndarray]:
    """Vértices y caras de un PLY binario (x y z [nx ny nz] r g b + caras con lista uchar)."""
    crudo = ruta.read_bytes()
    fin = crudo.index(b"\n", crudo.index(b"end_header")) + 1
    cab = crudo[:fin].decode(errors="replace").splitlines()
    nv = int(next(l for l in cab if l.startswith("element vertex")).split()[-1])
    nf = int(next(l for l in cab if l.startswith("element face")).split()[-1])
    tipos = {"float": "f4", "uchar": "u1", "double": "f8", "int": "i4", "uint": "u4"}
    props, lista = [], None
    en_vertices = False
    for l in cab:
        if l.startswith("element vertex"):
            en_vertices = True
        elif l.startswith("element"):
            en_vertices = False
        elif l.startswith("property list") and not en_vertices:
            lista = tipos[l.split()[3]]
        elif l.startswith("property") and en_vertices:
            _, t, nombre = l.split()
            props.append((nombre, tipos[t]))
    vt = np.dtype(props)
    v = np.frombuffer(crudo, vt, nv, fin)
    caras = np.frombuffer(crudo, np.dtype([("n", "u1"), ("i", "<" + lista, 3)]), nf, fin + nv * vt.itemsize)["i"]
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), caras.astype(np.int64)


def caras_orientadas(v: np.ndarray, f: np.ndarray):
    a, b, c = v[f[:, 0]], v[f[:, 1]], v[f[:, 2]]
    n = np.cross(b - a, c - a)
    area = np.linalg.norm(n, axis=1) / 2
    ok = area > 1e-9
    return (a + b + c)[ok] / 3, n[ok] / (2 * area[ok, None]), area[ok]


def pico(valores: np.ndarray, pesos: np.ndarray, paso: float = 0.01):
    """Histograma ponderado: centros y pesos de cada intervalo."""
    lo, hi = valores.min(), valores.max()
    bins = max(1, int(math.ceil((hi - lo) / paso)))
    h, bordes = np.histogram(valores, bins=bins, range=(lo, lo + bins * paso), weights=pesos)
    return (bordes[:-1] + bordes[1:]) / 2, h


def refinar(valores: np.ndarray, pesos: np.ndarray, centro: float, margen: float = 0.03) -> float:
    sel = np.abs(valores - centro) < margen
    return float(np.average(valores[sel], weights=pesos[sel])) if sel.any() else centro


def planos(v: np.ndarray, f: np.ndarray, registro=print) -> dict:
    """Suelo, techo y las cuatro paredes que cierran la habitación (en el mundo de ARKit: Y arriba)."""
    centro, n, area = caras_orientadas(v, f)

    hor_up, hor_down = n[:, 1] > 0.9, n[:, 1] < -0.9
    c, h = pico(centro[hor_up, 1], area[hor_up])
    # Suelo: el pico fuerte más bajo (no la tapa de un mueble). «Fuerte» = al menos el 30 % del mayor.
    suelo = refinar(centro[hor_up, 1], area[hor_up], c[np.argmax(h >= 0.3 * h.max())])
    c, h = pico(centro[hor_down, 1], area[hor_down])
    techo = refinar(centro[hor_down, 1], area[hor_down], c[len(h) - 1 - np.argmax((h >= 0.3 * h.max())[::-1])])

    # Paredes: caras verticales. Su dirección dominante (módulo 90°) da el giro de la habitación.
    vert = np.abs(n[:, 1]) < 0.15
    ang = np.degrees(np.arctan2(n[vert, 2], n[vert, 0])) % 90
    c, h = pico(ang, area[vert], 0.5)
    giro = math.radians(refinar(ang, area[vert], c[np.argmax(h)], 2.0))
    ejes = [np.array([math.cos(giro), 0, math.sin(giro)]), np.array([-math.sin(giro), 0, math.cos(giro)])]

    # Contorno de la habitación: lo que ocupa el suelo (percentiles, para que un resto suelto no lo agrande). Cada
    # pared es el plano vertical fuerte más cercano a ese borde; si no hay ninguno claro, el propio borde del suelo.
    en_suelo = hor_up & (np.abs(centro[:, 1] - suelo) < 0.05)
    limites = []
    for k, eje in enumerate(ejes):
        pie = centro[en_suelo] @ eje
        lo, hi = np.percentile(pie, 0.5), np.percentile(pie, 99.5)
        alineadas = vert & (np.abs(n @ eje) > 0.9) & (centro[:, 1] > suelo + 0.1) & (centro[:, 1] < techo - 0.1)
        d = centro[alineadas] @ eje
        c, h = pico(d, area[alineadas], 0.02)
        h = np.convolve(h, [0.25, 0.5, 0.25], mode="same")
        maximos = [i for i in range(1, len(h) - 1) if h[i] >= h[i - 1] and h[i] >= h[i + 1] and h[i] >= 0.05]
        def pared(borde):
            cerca = [i for i in maximos if abs(c[i] - borde) < 0.25]
            if not cerca:
                return borde, "borde del suelo"
            i = max(cerca, key=lambda j: h[j])
            return refinar(d, area[alineadas], c[i]), f"pared detectada ({h[i]:.2f} m² en 2 cm)"
        (a, ta), (b, tb) = pared(lo), pared(hi)
        registro(f"  eje {k}: suelo de {lo:.2f} a {hi:.2f} m → paredes en {a:.3f} ({ta}) y {b:.3f} ({tb})")
        limites.append((a, b))
    registro(f"Suelo y = {suelo:.3f} m, techo y = {techo:.3f} m (alto {techo - suelo:.2f} m); giro {math.degrees(giro):.1f}°")
    return {"suelo": suelo, "techo": techo, "giro": giro, "ejes": ejes, "limites": limites}


def caja(p: dict) -> list[dict]:
    """Seis rectángulos (centro, eje u, eje v, ancho, alto, normal hacia dentro) de la caja de la habitación."""
    e0, e1 = p["ejes"]
    (a0, b0), (a1, b1) = p["limites"]
    y0, y1 = p["suelo"], p["techo"]
    up = np.array([0.0, 1.0, 0.0])
    cen = lambda s0, s1, y: e0 * s0 + e1 * s1 + up * y  # noqa: E731
    largo, ancho, alto = b0 - a0, b1 - a1, y1 - y0
    m0, m1, my = (a0 + b0) / 2, (a1 + b1) / 2, (y0 + y1) / 2
    return [
        {"nombre": "suelo", "centro": cen(m0, m1, y0), "u": e0, "v": e1, "ancho": largo, "alto": ancho, "n": up},
        {"nombre": "techo", "centro": cen(m0, m1, y1), "u": e0, "v": -e1, "ancho": largo, "alto": ancho, "n": -up},
        {"nombre": "pared_1", "centro": cen(a0, m1, my), "u": -e1, "v": up, "ancho": ancho, "alto": alto, "n": e0},
        {"nombre": "pared_2", "centro": cen(b0, m1, my), "u": e1, "v": up, "ancho": ancho, "alto": alto, "n": -e0},
        {"nombre": "pared_3", "centro": cen(m0, a1, my), "u": e0, "v": up, "ancho": largo, "alto": alto, "n": e1},
        {"nombre": "pared_4", "centro": cen(m0, b1, my), "u": -e0, "v": up, "ancho": largo, "alto": alto, "n": -e1},
    ]


def contorno(paredes: list[dict]) -> list[list[float]] | None:
    """Polígono del suelo (x, z) encadenando los extremos de las paredes de RoomPlan; None si no cierra."""
    tramos = [(np.array(w["a"]), np.array(w["b"])) for w in paredes]
    a, b = tramos.pop(0)
    poli, actual = [a], b
    while tramos:
        dist = [min(np.linalg.norm(p - actual), np.linalg.norm(q - actual)) for p, q in tramos]
        i = int(np.argmin(dist))
        if dist[i] > 0.35:
            return None
        p, q = tramos.pop(i)
        if np.linalg.norm(q - actual) < np.linalg.norm(p - actual):
            p, q = q, p
        poli.append((p + actual) / 2)
        actual = q
    if np.linalg.norm(actual - poli[0]) > 0.35:
        return None
    poli[0] = (poli[0] + actual) / 2
    return [[round(float(x), 4), round(float(z), 4)] for x, z in poli]


def planos_roomplan(origen: Path, registro=print):
    """Envolvente con las paredes que midió RoomPlan en el iPhone (centro, normal hacia dentro, ancho y alto exactos).
    Es la medida más fiable: no depende de la malla del PC, que en una sala abierta o con escaleras confunde planos."""
    ruta = origen / "habitacion.json"
    if not ruta.exists():
        return None
    sala = json.loads(ruta.read_text("utf-8"))
    walls = [w for w in sala.get("walls", []) if w.get("width", 0) > 0.05]
    if len(walls) < 3:
        return None
    up = np.array([0.0, 1.0, 0.0])
    suelo = float(sala.get("floorY", min(w["center"][1] - w["height"] / 2 for w in walls)))
    techo = float(np.median([w["center"][1] + w["height"] / 2 for w in walls]))
    paredes, extremos = [], []
    for i, w in enumerate(sorted(walls, key=lambda w: -w["width"])):
        n = np.array(w["normal"], float)
        n[1] = 0
        n /= np.linalg.norm(n)
        u = np.cross(up, n)
        c = np.array(w["center"], float)
        paredes.append({"nombre": f"pared_{i + 1}", "centro": c, "u": u, "v": up, "ancho": float(w["width"]),
                        "alto": float(w["height"]), "n": n})
        a_, b_ = c - u * w["width"] / 2, c + u * w["width"] / 2
        extremos.append({"a": [a_[0], a_[2]], "b": [b_[0], b_[2]]})
    # Suelo y techo: rectángulo orientado como la pared más larga que abarca todas las paredes.
    e0 = paredes[0]["u"]
    e1 = np.cross(e0, up)
    pts = np.array([p["a"] for p in extremos] + [p["b"] for p in extremos])
    p3 = np.stack([pts[:, 0], np.zeros(len(pts)), pts[:, 1]], 1)
    s0, s1 = p3 @ e0, p3 @ e1
    m0, m1 = (s0.min() + s0.max()) / 2, (s1.min() + s1.max()) / 2
    largo, ancho = float(s0.max() - s0.min()), float(s1.max() - s1.min())
    cen = lambda y: e0 * m0 + e1 * m1 + up * y  # noqa: E731
    rects = [{"nombre": "suelo", "centro": cen(suelo), "u": e0, "v": e1, "ancho": largo, "alto": ancho, "n": up},
             {"nombre": "techo", "centro": cen(techo), "u": e0, "v": -e1, "ancho": largo, "alto": ancho, "n": -up}]
    poli = contorno(extremos)
    registro(f"RoomPlan: {len(walls)} paredes, suelo y = {suelo:.3f} m, techo y = {techo:.3f} m, "
             f"{'contorno cerrado' if poli else 'sin contorno cerrado (suelo rectangular)'}")
    return {"suelo": suelo, "techo": techo, "giro": math.atan2(e0[2], e0[0])}, rects + paredes, poli


# ----------------------------------------------------------------------------------------------- textura

def cargar_fotos(origen: Path, dev):
    fotos = []
    for sub in ("fotos_clave", "fotogramas"):
        indice = origen / sub / "camaras.json"
        if not indice.exists():
            continue
        for f in json.loads(indice.read_text("utf-8"))["fotos"]:
            info = f.get("profundidad")
            if not isinstance(info, dict):
                continue
            prof = leer_profundidad(origen / sub / info["archivo"], info)
            if prof is None:
                continue
            img = Image.open(origen / sub / f["archivo"]).convert("RGB")
            escala = min(1.0, 1600 / img.width)
            if escala < 1:
                img = img.resize((round(img.width * escala), round(img.height * escala)), Image.LANCZOS)
            sx, sy = img.width / f["ancho"], img.height / f["alto"]
            fotos.append({
                "img": torch.from_numpy(np.asarray(img).copy()).to(dev).float() / 255,
                "K": (f["fx"] * sx, f["fy"] * sy, f["cx"] * sx, f["cy"] * sy),
                "prof": torch.from_numpy(prof.mapa.copy()).to(dev),
                "Kp": (prof.fx, prof.fy, prof.cx, prof.cy),
                "w2c": torch.from_numpy(arkit_a_opencv(matriz_columnas(f["cameraToWorld"]))).to(dev).float(),
            })
    return fotos


def ganancias(fotos) -> list[torch.Tensor]:
    """Exposición de cada foto respecto a la mediana (ARKit no fija la exposición): ganancia por canal."""
    medias = torch.stack([f["img"].reshape(-1, 3).median(0).values for f in fotos])
    ref = medias.median(0).values
    return [(ref / m.clamp(min=1e-3)).clamp(0.6, 1.6) for m in medias]


def rellenar(color: torch.Tensor, peso: torch.Tensor) -> torch.Tensor:
    """Pirámide «push-pull»: los huecos toman el color medio de su entorno, a cualquier escala."""
    niveles = [(color * peso[..., None], peso)]
    while min(niveles[-1][1].shape) > 2:
        c, p = niveles[-1]
        niveles.append((F.avg_pool2d(c.permute(2, 0, 1)[None], 2, ceil_mode=True)[0].permute(1, 2, 0),
                        F.avg_pool2d(p[None, None], 2, ceil_mode=True)[0, 0]))
    c, p = niveles[-1]
    lleno = c / p.clamp(min=1e-6)[..., None]
    for c, p in reversed(niveles[:-1]):
        arriba = F.interpolate(lleno.permute(2, 0, 1)[None], size=p.shape, mode="bilinear", align_corners=False)[0]
        propio = c / p.clamp(min=1e-6)[..., None]
        alfa = p.clamp(max=1)[..., None]
        lleno = propio * alfa + arriba.permute(1, 2, 0) * (1 - alfa)
    return lleno


@torch.no_grad()
def textura(r: dict, fotos, gan, mm_px: float, dev, top: int = 3):
    W, H = max(8, round(r["ancho"] * 1000 / mm_px)), max(8, round(r["alto"] * 1000 / mm_px))
    s = (torch.arange(W, device=dev) + 0.5) / W - 0.5
    t = 0.5 - (torch.arange(H, device=dev) + 0.5) / H
    S, T = torch.meshgrid(s, t, indexing="xy")
    c = torch.tensor(r["centro"], device=dev).float()
    u = torch.tensor(r["u"], device=dev).float()
    v = torch.tensor(r["v"], device=dev).float()
    nrm = torch.tensor(r["n"], device=dev).float()
    P = (c + S[..., None] * u * r["ancho"] + T[..., None] * v * r["alto"]).reshape(-1, 3)
    mejores_p = torch.zeros(len(P), top, device=dev)
    mejores_c = torch.zeros(len(P), top, 3, device=dev)
    for f, g in zip(fotos, gan):
        R, tt = f["w2c"][:3, :3], f["w2c"][:3, 3]
        cam = P @ R.T + tt
        z = cam[:, 2]
        ok = z > 0.2
        fx, fy, cx, cy = f["K"]
        uu, vv = cam[:, 0] / z.clamp(min=1e-3) * fx + cx, cam[:, 1] / z.clamp(min=1e-3) * fy + cy
        h, w = f["img"].shape[:2]
        ok &= (uu >= 0) & (uu < w - 1) & (vv >= 0) & (vv < h - 1)
        # El LiDAR de esta foto debe ver justo el plano: si mide menos, hay algo delante (un mueble).
        pfx, pfy, pcx, pcy = f["Kp"]
        ph, pw = f["prof"].shape
        pu = (cam[:, 0] / z.clamp(min=1e-3) * pfx + pcx).long().clamp(0, pw - 1)
        pv = (cam[:, 1] / z.clamp(min=1e-3) * pfy + pcy).long().clamp(0, ph - 1)
        d = f["prof"][pv, pu]
        ok &= (d > 0) & ((d - z).abs() < 0.04 + 0.01 * z)
        # Puntuación: de frente y de cerca (el detalle por píxel cae con la distancia y con el ángulo).
        centro_cam = -R.T @ tt
        rayo = F.normalize(centro_cam - P, dim=1)
        frente = (rayo @ nrm).clamp(min=0)
        p = torch.where(ok & (frente > 0.25), frente ** 2 / z.clamp(min=0.3) ** 2, torch.zeros_like(z))
        if not (p > 0).any():
            continue
        col = F.grid_sample(f["img"].permute(2, 0, 1)[None],
                            torch.stack([uu / (w - 1) * 2 - 1, vv / (h - 1) * 2 - 1], 1)[None, None],
                            mode="bilinear", align_corners=True)[0, :, 0].T * g
        # Se guardan las `top` mejores fotos por punto (la mezcla de varias suaviza costuras sin emborronar).
        todas_p = torch.cat([mejores_p, p[:, None]], 1)
        todas_c = torch.cat([mejores_c, col[:, None]], 1)
        orden = todas_p.argsort(1, descending=True)[:, :top]
        mejores_p = todas_p.gather(1, orden)
        mejores_c = todas_c.gather(1, orden[..., None].expand(-1, -1, 3))
    peso = mejores_p ** 2
    total = peso.sum(1)
    color = (mejores_c * peso[..., None]).sum(1) / total.clamp(min=1e-12)[:, None]
    visto = (total > 0).float()
    cobertura = float(visto.mean())
    # Color real del acabado (pintura, baldosa…): mediana de lo visto limpio, sin sombras ni muebles.
    base = color[total > 0].median(0).values if (total > 0).any() else torch.full((3,), 0.8, device=dev)
    color = rellenar(color.reshape(H, W, 3), visto.reshape(H, W))
    return (color.clamp(0, 1) * 255).byte().cpu().numpy(), cobertura, [round(float(x), 4) for x in base]


# ----------------------------------------------------------------------------------------------- salida

def escribir_obj(destino: Path, rects: list[dict]):
    obj, mtl = ["mtllib envolvente.mtl"], []
    k = 0
    for r in rects:
        c, u, v = np.array(r["centro"]), np.array(r["u"]) * r["ancho"] / 2, np.array(r["v"]) * r["alto"] / 2
        obj.append(f"o {r['nombre']}")
        for p in (c - u - v, c + u - v, c + u + v, c - u + v):
            obj.append(f"v {p[0]:.5f} {p[1]:.5f} {p[2]:.5f}")
        obj += ["vt 0 0", "vt 1 0", "vt 1 1", "vt 0 1"]
        n = np.array(r["n"])
        obj.append(f"vn {n[0]:.5f} {n[1]:.5f} {n[2]:.5f}")
        obj.append(f"usemtl {r['nombre']}")
        # Orden de los vértices para que la cara mire hacia dentro de la habitación.
        orden = (1, 2, 3, 4) if np.dot(np.cross(u, v), n) > 0 else (1, 4, 3, 2)
        obj.append("f " + " ".join(f"{k + i}/{k + i}/{k // 4 + 1}" for i in orden))
        k += 4
        mtl += [f"newmtl {r['nombre']}", "Kd 1 1 1", "Ks 0 0 0", "illum 1", f"map_Kd {r['nombre']}.png", ""]
    (destino / "envolvente.obj").write_text("\n".join(obj) + "\n")
    (destino / "envolvente.mtl").write_text("\n".join(mtl), "utf-8")


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    p.add_argument("--mm-px", type=float, default=3.0, help="tamaño de cada píxel de textura en milímetros")
    a = p.parse_args()
    origen = Path(a.carpeta).resolve()
    destino = origen / "remodelado"
    destino.mkdir(exist_ok=True)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # La malla refinada del PC es lisa; la del LiDAR sale en escalones de vóxel y sus normales engañan.
    # Con RoomPlan (habitacion.json) las paredes vienen medidas del iPhone; si no, se buscan en la malla.
    roomplan = planos_roomplan(origen)
    if roomplan:
        pl, rects, poli = roomplan
    else:
        lisa = origen / "splat2d" / "malla_2dgs_plana.ply"
        v, f = leer_malla(lisa if lisa.exists() else origen / "malla_color.ply")
        pl, poli = planos(v, f), None
        rects = caja(pl)
    fotos = cargar_fotos(origen, dev)
    gan = ganancias(fotos)
    print(f"{len(fotos)} fotos con LiDAR", flush=True)
    medidas = {}
    for r in rects:
        img, cobertura, base = textura(r, fotos, gan, a.mm_px, dev)
        Image.fromarray(img).save(destino / f"{r['nombre']}.png", optimize=True)
        medidas[r["nombre"]] = {"ancho_m": round(r["ancho"], 3), "alto_m": round(r["alto"], 3),
                                "visto_limpio": round(cobertura, 3), "color": base,
                                "u": [round(x, 4) for x in r["u"]], "v": [round(x, 4) for x in r["v"]],
                                "centro": [round(x, 4) for x in r["centro"]], "normal": [round(x, 4) for x in r["n"]]}
        print(f"  {r['nombre']}: {r['ancho']:.2f} × {r['alto']:.2f} m, visto sin obstáculos el {cobertura * 100:.0f} %",
              flush=True)
    escribir_obj(destino, rects)
    resumen = {"unidades": "metros, mundo de ARKit (Y arriba)",
               "largo_m": round(rects[0]["ancho"], 3), "ancho_m": round(rects[0]["alto"], 3),
               "alto_m": round(pl["techo"] - pl["suelo"], 3), "suelo_y": round(pl["suelo"], 4),
               "techo_y": round(pl["techo"], 4), "giro_grados": round(math.degrees(pl["giro"]), 2),
               "fuente": "RoomPlan" if roomplan else "malla", "contorno_xz": poli, "planos": medidas}
    (destino / "envolvente.json").write_text(json.dumps(resumen, indent=1, ensure_ascii=False), "utf-8")
    print(f"Envolvente: {resumen['largo_m']} × {resumen['ancho_m']} × {resumen['alto_m']} m → {destino}", flush=True)


if __name__ == "__main__":
    main()
