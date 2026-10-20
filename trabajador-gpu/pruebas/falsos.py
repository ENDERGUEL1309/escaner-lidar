"""Escena sintética y modelos falsos para probar el proceso sin GPU ni pesos de IA.

La escena imita un escaneo del iPhone: suelo y paredes (estructura) y una caja roja (silla) pegada a una
caja azul (mesa) que el iPhone dejó como un único objeto. Los modelos falsos "detectan" por color.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escaner_ia import ply  # noqa: E402

PASO = 0.02
ROJO, AZUL, SUELO, PARED = (200, 40, 40), (40, 60, 200), (150, 140, 120), (220, 220, 210)


def rectangulo(origen, eje_u, eje_v, normal, color):
    origen, eje_u, eje_v = (np.asarray(x, np.float32) for x in (origen, eje_u, eje_v))
    nu = max(2, int(round(np.linalg.norm(eje_u) / PASO)) + 1)
    nv = max(2, int(round(np.linalg.norm(eje_v) / PASO)) + 1)
    a, b = np.meshgrid(np.linspace(0, 1, nu), np.linspace(0, 1, nv), indexing="ij")
    pos = origen + a.reshape(-1, 1) * eje_u + b.reshape(-1, 1) * eje_v
    i = np.arange(nu * nv).reshape(nu, nv)
    q = np.stack([i[:-1, :-1], i[1:, :-1], i[1:, 1:], i[:-1, 1:]], -1).reshape(-1, 4)
    caras = np.concatenate([q[:, [0, 1, 2]], q[:, [0, 2, 3]]])
    if np.dot(np.cross(eje_u, eje_v), normal) < 0:
        caras = caras[:, [0, 2, 1]]
    ruido = np.random.default_rng(len(pos)).integers(-12, 12, (len(pos), 3))
    return ply.Malla(posiciones=pos.astype(np.float32), normales=np.tile(np.float32(normal), (len(pos), 1)),
                     colores=np.clip(np.asarray(color) + ruido, 0, 255).astype(np.uint8),
                     caras=caras.astype(np.int64),
                     st=np.column_stack([pos[:, 0] * 0.1 + 0.5, pos[:, 2] * 0.1 + 0.5]).astype(np.float32))


def caja(centro, tam, color):
    cx, cy, cz = centro
    hx, hy, hz = (t / 2 for t in tam)
    x0, x1, y0, y1, z0, z1 = cx - hx, cx + hx, cy - hy, cy + hy, cz - hz, cz + hz
    return ply.unir([
        rectangulo((x0, y1, z0), (2 * hx, 0, 0), (0, 0, 2 * hz), (0, 1, 0), color),
        rectangulo((x1, y0, z0), (0, 2 * hy, 0), (0, 0, 2 * hz), (1, 0, 0), color),
        rectangulo((x0, y0, z0), (0, 2 * hy, 0), (0, 0, 2 * hz), (-1, 0, 0), color),
        rectangulo((x0, y0, z1), (2 * hx, 0, 0), (0, 2 * hy, 0), (0, 0, 1), color),
        rectangulo((x0, y0, z0), (2 * hx, 0, 0), (0, 2 * hy, 0), (0, 0, -1), color),
    ])[0]


def crear_escaneo(carpeta: Path) -> Path:
    from PIL import Image

    carpeta.mkdir(parents=True, exist_ok=True)
    (carpeta / "objetos").mkdir(exist_ok=True)
    suelo = rectangulo((-1.5, 0, -1.5), (3, 0, 0), (0, 0, 3), (0, 1, 0), SUELO)
    paredes = ply.unir([
        rectangulo((-1.5, 0, -1.5), (3, 0, 0), (0, 2.4, 0), (0, 0, 1), PARED),
        rectangulo((-1.5, 0, 1.5), (3, 0, 0), (0, 2.4, 0), (0, 0, -1), PARED),
        rectangulo((-1.5, 0, -1.5), (0, 0, 3), (0, 2.4, 0), (1, 0, 0), PARED),
        rectangulo((1.5, 0, -1.5), (0, 0, 3), (0, 2.4, 0), (-1, 0, 0), PARED),
    ])[0]
    juntos = ply.unir([caja((-0.25, 0.25, 0.4), (0.5, 0.5, 0.5), ROJO), caja((0.3, 0.35, 0.4), (0.6, 0.7, 0.6), AZUL)])[0]
    partes = [("Suelo", "suelo", True, suelo), ("Paredes", "pared", True, paredes), ("Objeto 1", "objeto", False, juntos)]

    objetos = []
    for nombre, cat, estructura, m in partes:
        archivo = f"objetos/{nombre.replace(' ', '_')}.ply"
        m.textura = "../textura.jpg"
        ply.escribir(m, carpeta / archivo, inferido=False)
        objetos.append({"nombre": nombre, "categoria": cat, "estructura": estructura, "centro": [0, 0, 0],
                        "dimensiones": [1, 1, 1], "rotacionY": 0, "vertices": m.n_vertices,
                        "triangulos": len(m.caras), "archivo": archivo})
    completa, _ = ply.unir([m for *_, m in partes])
    completa.st = None
    ply.escribir(completa, carpeta / "malla_color.ply", inferido=False)
    (carpeta / "escena.json").write_text(json.dumps({"version": 1, "unidades": "metros", "ejeVertical": "Y",
                                                     "objetos": objetos}, indent=2), "utf-8")
    Image.new("RGB", (64, 64), (128, 128, 128)).save(carpeta / "textura.jpg")
    return carpeta


def _color(imagen, color, tol=45):
    return np.all(np.abs(imagen.astype(int) - np.asarray(color)) < tol, axis=-1)


class DetectorFalso:
    def detectar(self, imagen, frases):
        from scipy import ndimage

        salida = []
        for color, frase in ((ROJO, "chair"), (AZUL, "table")):
            if frase not in frases:
                continue
            etiquetas, n = ndimage.label(_color(imagen, color))
            for k, sl in enumerate(ndimage.find_objects(etiquetas)):
                if (etiquetas[sl] == k + 1).sum() < 200:
                    continue
                y, x = sl
                salida.append((np.array([x.start, y.start, x.stop, y.stop], np.float32), 0.8, frases.index(frase)))
        return salida


class SegmentadorFalso:
    def segmentar(self, imagen, cajas):
        mascaras = np.zeros((len(cajas),) + imagen.shape[:2], bool)
        for i, (x0, y0, x1, y1) in enumerate(cajas.astype(int)):
            trozo = imagen[y0:y1, x0:x1]
            color = ROJO if _color(trozo, ROJO).sum() > _color(trozo, AZUL).sum() else AZUL
            mascaras[i, y0:y1, x0:x1] = _color(trozo, color)
        return mascaras


class NombradorFalso:
    escala = 50.0

    def vectores_imagen(self, imagenes):
        v = np.array([np.asarray(im, float).reshape(-1, 3).mean(0) - 128 for im in imagenes])
        v = np.pad(v, ((0, 0), (0, 125)))
        return v / np.linalg.norm(v, axis=1, keepdims=True)

    def vectores_texto(self, textos):
        v = np.zeros((len(textos), 128))
        for i, t in enumerate(textos):
            if t == "a photo of a chair.":
                v[i, :3] = np.asarray(ROJO) - 128
            elif t == "a photo of a table.":
                v[i, :3] = np.asarray(AZUL) - 128
            else:
                v[i, 3 + i % 125] = 1
        return v / np.linalg.norm(v, axis=1, keepdims=True)


class ModelosFalsos:
    detector, segmentador, nombrador = DetectorFalso(), SegmentadorFalso(), NombradorFalso()
    versiones = ["falso"]
