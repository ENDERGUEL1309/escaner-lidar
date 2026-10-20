"""Vistas (foto + cámara) en convención OpenCV: x derecha, y abajo, z hacia delante.

La app guarda `cameraToWorld` de ARKit (x derecha, y arriba, z hacia atrás) por columnas, como
`simd_float4x4` (ver docs/ia-nube.md §3). Aquí se invierte y se voltean Y y Z.
"""

from __future__ import annotations

import json
import math
from dataclasses import dataclass
from pathlib import Path

import numpy as np

ARKIT_A_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])


@dataclass
class Profundidad:
    mapa: np.ndarray            # (h, w) float32, metros; 0 o nan = sin dato
    fx: float
    fy: float
    cx: float
    cy: float


@dataclass
class Vista:
    nombre: str
    imagen: np.ndarray          # (H, W, 3) uint8, ya reescalada a la resolución de trabajo
    fx: float
    fy: float
    cx: float
    cy: float
    mundo_a_cam: np.ndarray     # (4, 4) OpenCV
    profundidad: Profundidad | None = None
    tiempo: float = 0.0
    sintetica: bool = False

    @property
    def ancho(self) -> int:
        return self.imagen.shape[1]

    @property
    def alto(self) -> int:
        return self.imagen.shape[0]

    @property
    def posicion(self) -> np.ndarray:
        return np.linalg.inv(self.mundo_a_cam)[:3, 3]

    def a_camara(self, puntos: np.ndarray) -> np.ndarray:
        r, t = self.mundo_a_cam[:3, :3], self.mundo_a_cam[:3, 3]
        return (puntos @ r.T.astype(np.float32)) + t.astype(np.float32)

    def proyectar(self, puntos: np.ndarray):
        """→ (u, v, z) en píxeles de la imagen de trabajo; z en metros (≤ 0 detrás de la cámara)."""
        pc = self.a_camara(puntos)
        z = pc[:, 2]
        zs = np.where(z > 1e-6, z, 1e-6)
        u = self.fx * pc[:, 0] / zs + self.cx
        v = self.fy * pc[:, 1] / zs + self.cy
        return u, v, z


def matriz_columnas(valores) -> np.ndarray:
    """Lista de 16 números por columnas (simd) → matriz 4×4."""
    return np.asarray(valores, np.float64).reshape(4, 4).T


def arkit_a_opencv(camera_to_world: np.ndarray) -> np.ndarray:
    return ARKIT_A_OPENCV @ np.linalg.inv(camera_to_world)


def leer_profundidad(ruta: Path, info: dict) -> Profundidad | None:
    """`.prof`: cabecera de 8 bytes (ancho, alto como uint32) + Float16 ancho×alto [+ confianza uint8 ancho×alto]."""
    try:
        datos = ruta.read_bytes()
        w, h = (int(x) for x in np.frombuffer(datos, "<u4", 2, 0))
        mapa = np.frombuffer(datos, "<f2", w * h, 8).astype(np.float32).reshape(h, w)
        resto = len(datos) - 8 - 2 * w * h
        if resto >= w * h:   # confianza de ARKit: 0 baja, 1 media, 2 alta
            conf = np.frombuffer(datos, "u1", w * h, 8 + 2 * w * h).reshape(h, w)
            mapa = np.where(conf >= 1, mapa, 0)
    except Exception:
        return None
    ancho0, alto0 = info.get("ancho", w), info.get("alto", h)
    sx, sy = w / ancho0, h / alto0     # por si el mapa se guardó a otro tamaño que el de sus intrínsecos
    return Profundidad(np.nan_to_num(mapa, nan=0.0), info["fx"] * sx, info["fy"] * sy, info["cx"] * sx, info["cy"] * sy)


def cargar_fotos(carpeta: Path, lado_max: int = 1600) -> tuple[list[Vista], list[str]]:
    """Lee `fotos_clave/camaras.json` y sus fotos. Devuelve las vistas y los avisos."""
    from PIL import Image

    avisos: list[str] = []
    info = json.loads((carpeta / "camaras.json").read_text("utf-8"))
    if info.get("convencion", "arkit") != "arkit":
        raise ValueError(f"camaras.json: convención no admitida: {info.get('convencion')}")
    vistas = []
    for foto in sorted(info.get("fotos", []), key=lambda f: f.get("marcaTiempo", 0)):
        ruta = carpeta / foto["archivo"]
        if not ruta.exists():
            avisos.append(f"Falta la foto {foto['archivo']}")
            continue
        imagen = Image.open(ruta).convert("RGB")       # sin aplicar EXIF: los intrínsecos son del sensor
        ancho0, alto0 = foto.get("ancho", imagen.width), foto.get("alto", imagen.height)
        if abs(imagen.width / imagen.height - ancho0 / alto0) > 0.02:
            avisos.append(f"{foto['archivo']}: la foto mide {imagen.width}×{imagen.height} y camaras.json dice "
                          f"{ancho0}×{alto0}; se omite")
            continue
        escala = min(1.0, lado_max / max(imagen.width, imagen.height))
        if escala < 1:
            imagen = imagen.resize((round(imagen.width * escala), round(imagen.height * escala)), Image.BILINEAR)
        sx, sy = imagen.width / ancho0, imagen.height / alto0
        prof = None
        if isinstance(foto.get("profundidad"), dict):
            prof = leer_profundidad(carpeta / foto["profundidad"]["archivo"], foto["profundidad"])
        vistas.append(Vista(
            nombre=foto["archivo"],
            imagen=np.asarray(imagen),
            fx=foto["fx"] * sx, fy=foto["fy"] * sy,
            # +0.5/−0.5: los intrínsecos se refieren a centros de píxel en la resolución original.
            cx=(foto["cx"] + 0.5) * sx - 0.5, cy=(foto["cy"] + 0.5) * sy - 0.5,
            mundo_a_cam=arkit_a_opencv(matriz_columnas(foto["cameraToWorld"])),
            profundidad=prof,
            tiempo=float(foto.get("marcaTiempo", 0)),
        ))
    return vistas, avisos


def mirar_a(origen, objetivo, arriba=(0.0, 1.0, 0.0)) -> np.ndarray:
    """Matriz mundo→cámara (OpenCV) de una cámara en `origen` que mira a `objetivo` con el mundo Y arriba."""
    origen, objetivo = np.asarray(origen, float), np.asarray(objetivo, float)
    z = objetivo - origen
    z /= np.linalg.norm(z)
    x = np.cross(z, np.asarray(arriba, float))
    if np.linalg.norm(x) < 1e-6:
        x = np.cross(z, [0.0, 0.0, 1.0])
    x /= np.linalg.norm(x)
    y = np.cross(z, x)                     # y abajo
    r = np.stack([x, y, z])                # filas = ejes de la cámara en el mundo
    m = np.eye(4)
    m[:3, :3] = r
    m[:3, 3] = -r @ origen
    return m


def camaras_virtuales(posiciones: np.ndarray, ancho=1280, alto=960, fov_h=75.0):
    """Cámaras para el modo de prueba sin fotos: desde el centro de la sala mirando alrededor
    (dos inclinaciones) y desde las esquinas mirando al centro. Devuelve [(nombre, fx, fy, cx, cy, mundo_a_cam)]."""
    p = posiciones
    suelo = float(np.percentile(p[:, 1], 2))
    techo = float(np.percentile(p[:, 1], 98))
    lo = np.percentile(p[:, [0, 2]], 3, axis=0)
    hi = np.percentile(p[:, [0, 2]], 97, axis=0)
    centro_xz = (lo + hi) / 2
    ojo = min(suelo + 1.4, (suelo + techo) / 2 + 0.3) if techo - suelo > 0.5 else suelo + 1.4
    centro = np.array([centro_xz[0], ojo, centro_xz[1]])
    f = (ancho / 2) / math.tan(math.radians(fov_h) / 2)

    camaras = []
    for inclinacion in (-20.0, -50.0):
        for k in range(12):
            yaw = 2 * math.pi * k / 12
            d = np.array([math.sin(yaw) * math.cos(math.radians(inclinacion)),
                          math.sin(math.radians(inclinacion)),
                          math.cos(yaw) * math.cos(math.radians(inclinacion))])
            camaras.append((f"virtual_c{int(-inclinacion)}_{k:02d}", centro, centro + d))
    mitad = (hi - lo) / 2 * 0.85
    objetivo = np.array([centro_xz[0], suelo + 0.6, centro_xz[1]])
    for i, (sx, sz) in enumerate(((-1, -1), (1, -1), (1, 1), (-1, 1))):
        esquina = np.array([centro_xz[0] + sx * mitad[0], ojo + 0.2, centro_xz[1] + sz * mitad[1]])
        base = objetivo - esquina
        for j, giro in enumerate((-30.0, 0.0, 30.0)):
            a = math.radians(giro)
            d = np.array([base[0] * math.cos(a) + base[2] * math.sin(a), base[1],
                          -base[0] * math.sin(a) + base[2] * math.cos(a)])
            camaras.append((f"virtual_e{i}_{j}", esquina, esquina + d))
    return [(n, f, f, ancho / 2 - 0.5, alto / 2 - 0.5, mirar_a(o, t)) for n, o, t in camaras]
