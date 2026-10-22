r"""Modelado verificado con fotos: cada mueble es un modelo paramétrico limpio que se ajusta hasta que, visto desde
la cámara real de cada foto, ocupa exactamente lo que ocupa el objeto real.

    splat2d_local.bat … (antes)   →   .venv-splat\Scripts\python modelado_verificado.py "C:\…\Entorno_…"

Para cada vista elegida:
  · profundidad de la escena: el modelo 2DGS renderizado desde esa cámara (lo que de verdad hay delante);
  · silueta de referencia: los píxeles cuya superficie 3D cae en la zona del objeto (no suelo ni paredes);
  · silueta del modelo: trazado de rayos contra las cajas del modelo, ocultando lo que el objeto real tiene delante.
La coincidencia es la IoU media de las siluetas; Nelder-Mead ajusta posición, giro y medidas. Deja superposiciones
sobre las fotos (verde: objeto real, rojo: modelo) en <escaneo>/modelado/ y los parámetros en modelado.json.

Coordenadas: las del mundo de Blender del proyecto (X, Y en planta, Z arriba; Blender = (x_a, −z_a, y_a) de ARKit).
"""

from __future__ import annotations

import argparse
import json
import math
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image, ImageDraw
from scipy.optimize import minimize
from scipy.spatial import cKDTree

import splat2d_local as s2d
from suavizar_malla import leer

RES = 0.25  # resolución de trabajo respecto a las imágenes de 1280 px


# ----------------------------------------------------------------------------------------------- modelos

def rot_z(a):
    c, s = math.cos(a), math.sin(a)
    return np.array([[c, -s, 0], [s, c, 0], [0, 0, 1.0]])


class Mueble:
    """Modelo paramétrico: unión de cajas en coordenadas locales (x largo, y fondo, z alto desde el suelo)."""

    nombres: list[str] = []

    def __init__(self, p0):
        self.p = np.array(p0, float)

    def cajas(self, p):  # -> lista de (min xyz, max xyz) locales
        raise NotImplementedError

    def pose(self, p):  # centro (x, y) en planta y giro
        return p[0], p[1], p[2]


class Cama(Mueble):
    """Canapé con colchón (una caja) y dos almohadas en la cabecera (extremo −x)."""
    nombres = ["x", "y", "giro", "largo", "ancho", "alto"]
    contra_pared = (-1, 0)   # la cabecera

    def cajas(self, p):
        _, _, _, L, W, H = p
        a, f, e = 0.14, 0.42, 0.04   # alto, fondo y separación de las almohadas
        return [
            (np.array([-L / 2, -W / 2, 0]), np.array([L / 2, W / 2, H])),
            (np.array([-L / 2 + e, -W / 2 + e, H]), np.array([-L / 2 + e + f, -e / 2, H + a])),
            (np.array([-L / 2 + e, e / 2, H]), np.array([-L / 2 + e + f, W / 2 - e, H + a])),
        ]

    def techo(self, p):  # altura de la superficie principal (lo de encima no es la cama)
        return p[5]


class Escritorio(Mueble):
    """Tablero, cajonero en un extremo (lado +x) y panel lateral en el otro; la espalda (+y) contra la pared.
    Sin faldón trasero: en este escaneo la prueba de espacio libre lo descartó (las cámaras ven a través)."""
    nombres = ["x", "y", "giro", "largo", "fondo", "alto", "ancho_cajonero"]
    contra_pared = (0, 1)    # la espalda
    lado = 1  # -1: cajonero en el extremo −x

    def techo(self, p):
        return p[5]

    def partes_aspecto(self, p):
        """(caja que debe verse como mueble, caja que debe verse como hueco): el cajonero frente al hueco de las
        piernas. En la foto un frente de cajones iluminado es claro y el hueco bajo el tablero, oscuro."""
        _, _, _, L, D, H, wc = p
        wc = float(np.clip(wc, 0.25, 0.55 * L))
        s = 1 if self.lado >= 0 else -1
        x = lambda a, b: (min(s * a, s * b), max(s * a, s * b))
        c0, c1 = x(L / 2 - wc, L / 2)
        h0, h1 = x(-L / 2 + 0.05, L / 2 - wc - 0.03)
        return ((np.array([c0, -D / 2, 0.1]), np.array([c1, D / 2, H - 0.05])),
                (np.array([h0, -D / 2, 0.1]), np.array([h1, D / 2, H - 0.08])))

    def cajas(self, p):
        _, _, _, L, D, H, wc = p
        wc = float(np.clip(wc, 0.25, 0.55 * L))   # un cajonero, no un bloque macizo: queda hueco para las piernas
        t = 0.04
        cajas = [
            (np.array([-L / 2, -D / 2, H - t]), np.array([L / 2, D / 2, H])),            # tablero
            (np.array([L / 2 - wc, -D / 2, 0]), np.array([L / 2, D / 2, H - t])),        # cajonero
            (np.array([-L / 2, -D / 2, 0]), np.array([-L / 2 + t, D / 2, H - t])),       # lateral
        ]
        if self.lado < 0:
            espejo = np.array([-1.0, 1, 1])
            cajas = [(np.minimum(lo * espejo, hi * espejo), np.maximum(lo * espejo, hi * espejo)) for lo, hi in cajas]
        return cajas


class MallaRigida(Mueble):
    """Un modelo detallado (p. ej. generado por IA) que solo se coloca: posición, giro y escala uniforme.
    Las pruebas usan sus vértices: silueta por proyección de puntos, 3D y espacio libre por vecindad."""
    nombres = ["x", "y", "giro", "escala"]
    contra_pared = None

    def __init__(self, p0, vertices, caras=None):
        super().__init__(p0)
        c = (vertices.min(0) + vertices.max(0)) / 2
        v = vertices - np.array([c[0], c[1], vertices[:, 2].min()])          # centrada en planta, apoyada en z=0
        if caras is not None:
            # Puntos también dentro de cada triángulo: la silueta proyectada queda llena, sin agujeros.
            t = v[caras]
            pesos = np.array([[1, 1, 1], [4, 1, 1], [1, 4, 1], [1, 1, 4], [1, 1, 0], [0, 1, 1], [1, 0, 1]], float)
            pesos /= pesos.sum(1, keepdims=True)
            v = np.concatenate([v, np.einsum("kj,tjd->tkd", pesos, t).reshape(-1, 3)])
        self.v = v
        rng = np.random.default_rng(0)
        self.muestra = self.v[rng.choice(len(self.v), min(4000, len(self.v)), replace=False)]

    @staticmethod
    def escala(p):
        return float(np.clip(p[3], 0.6, 1.6))

    def cajas(self, p):  # caja envolvente (para el encuadre y las paredes)
        e = self.escala(p)
        return [(self.v.min(0) * e, self.v.max(0) * e)]

    def techo(self, p):
        return 3.0   # no tiene una superficie con cosas encima

    def puntos_mundo(self, p, todos=False):
        x, y, giro = p[:3]
        e = self.escala(p)
        v = self.v if todos else self.muestra
        return (v * e) @ rot_z(giro).T + np.array([x, y, 0.0])


# ----------------------------------------------------------------------------------------------- vistas

def a_blender(p):  # ARKit (Y arriba) -> Blender (Z arriba)
    return np.stack([p[..., 0], -p[..., 2], p[..., 1]], -1)


class Vista:
    def __init__(self, v, prof, alfa):
        self.nombre = v["nombre"]
        self.img = v["img"].numpy()
        K = v["K"].copy()
        K[:2] *= RES
        self.K = K
        self.alto, self.ancho = prof.shape
        c2w = np.linalg.inv(v["w2c"])
        uu, vv = np.meshgrid(np.arange(self.ancho) + 0.5, np.arange(self.alto) + 0.5)
        d_cam = np.stack([(uu - K[0, 2]) / K[0, 0], (vv - K[1, 2]) / K[1, 1], np.ones_like(uu)], -1)
        # Rayos con componente z de cámara = 1: el parámetro t del rayo es la profundidad.
        self.origen = a_blender(c2w[:3, 3])
        self.dir = a_blender(d_cam @ c2w[:3, :3].T)
        # Ejes de la cámara en el mundo de Blender (para proyectar puntos).
        self.dir_x, self.dir_y, self.dir_z = (a_blender(c2w[:3, k]) for k in range(3))
        self.prof = np.where(alfa > 0.5, prof, 0.0)
        self.puntos = self.origen + self.dir * self.prof[..., None]

    def silueta_modelo(self, mueble, p, suelo):
        if isinstance(mueble, MallaRigida):
            return self.silueta_puntos(mueble.puntos_mundo(p, todos=True) + np.array([0, 0, suelo]))
        x, y, giro = mueble.pose(p)
        R = rot_z(-giro)
        o = R @ (self.origen - np.array([x, y, suelo]))
        d = self.dir @ R.T
        t_hit = np.full(self.prof.shape, np.inf)
        with np.errstate(divide="ignore", invalid="ignore"):
            inv = 1.0 / d
            for lo, hi in mueble.cajas(p):
                t1, t2 = (lo - o) * inv, (hi - o) * inv
                tmin = np.nanmax(np.minimum(t1, t2), -1)
                tmax = np.nanmin(np.maximum(t1, t2), -1)
                ok = (tmax >= np.maximum(tmin, 0.05))
                t_hit = np.where(ok & (tmin < t_hit), np.maximum(tmin, 0.05), t_hit)
        visible = np.isfinite(t_hit) & ((self.prof == 0) | (t_hit < self.prof + 0.04))
        return visible, t_hit

    def silueta_puntos(self, puntos):
        """Silueta de una malla densa proyectando sus vértices (z-buffer por píxel y cierre de huecos de 1 px)."""
        cam = (puntos - self.origen) @ np.stack([self.dir_x, self.dir_y, self.dir_z], 1)
        z = cam[:, 2]
        ok = z > 0.05
        u = (cam[ok, 0] / z[ok] * self.K[0, 0] + self.K[0, 2]).astype(int)
        w = (cam[ok, 1] / z[ok] * self.K[1, 1] + self.K[1, 2]).astype(int)
        dentro = (u >= 0) & (u < self.ancho) & (w >= 0) & (w < self.alto)
        prof = np.full(self.alto * self.ancho, np.inf)
        np.minimum.at(prof, w[dentro] * self.ancho + u[dentro], z[ok][dentro])
        prof = prof.reshape(self.alto, self.ancho)
        # Cierre: un píxel vacío rodeado de modelo toma la profundidad mínima de sus vecinos (dos pasadas).
        for _ in range(2):
            vecinos = np.min([np.roll(np.roll(prof, a, 0), b, 1) for a in (-1, 0, 1) for b in (-1, 0, 1)], 0)
            lleno = np.sum([np.isfinite(np.roll(np.roll(prof, a, 0), b, 1)) for a in (-1, 0, 1) for b in (-1, 0, 1)], 0)
            prof = np.where(np.isinf(prof) & (lleno >= 4), vecinos, prof)
        # Tolerancia amplia: un modelo de IA se parece al objeto, pero no es una copia exacta de su superficie.
        visible = np.isfinite(prof) & ((self.prof == 0) | (prof < self.prof + 0.12))
        return visible, prof


def cerca_de_pared(pared, q, margen=0.0):
    """¿Los puntos q (ejes de la habitación) caen frente al tramo de la pared?"""
    eje, pos, lo, hi = pared
    return (q[..., 1 - eje] > lo - margen) & (q[..., 1 - eje] < hi + margen)


def orientar(mueble, p, paredes, giro_hab):
    """Giro (p[2] o p[2] + π) con el que el lado `contra_pared` del mueble apunta a la pared más cercana."""
    if mueble.contra_pared is None:
        return p[2]
    centro = rot_z(-giro_hab)[:2, :2] @ p[:2]
    rel = p[2] - giro_hab
    direccion = rot_z(rel)[:2, :2] @ np.array(mueble.contra_pared)     # en ejes de la habitación
    eje = int(np.argmax(np.abs(direccion)))
    candidatas = [pared[1] for pared in paredes if pared[0] == eje and cerca_de_pared(pared, centro, 0.3)]
    if not candidatas:
        return p[2]
    pared = min(candidatas, key=lambda pos: abs(pos - centro[eje]))
    return p[2] if np.sign(pared - centro[eje]) == np.sign(direccion[eje]) else p[2] + math.pi


def zona(mueble, p, puntos, suelo, margen, paredes=(), giro_hab=0.0, margen_arriba=0.02):
    """Puntos de la escena del objeto y los neutros.

    Objeto: dentro de la caja envolvente del modelo ampliada `margen` en planta y hasta `margen_arriba` por encima de
    su parte más alta, sin suelo ni paredes. Neutros: lo que hay encima de la superficie principal del mueble
    (monitor, PC, almohadas, un gato…): tapa el mueble en la foto pero no es el mueble, así que no cuenta ni a favor
    ni en contra del modelo."""
    x, y, giro = mueble.pose(p)
    local = (puntos - np.array([x, y, suelo])) @ rot_z(-giro).T
    cajas = mueble.cajas(p)
    lo = np.min([c[0] for c in cajas], 0) - margen
    hi = np.max([c[1] for c in cajas], 0) + margen
    lo[2], hi[2] = 0.03, hi[2] - margen + margen_arriba
    en_planta = np.all((local[..., :2] > lo[:2]) & (local[..., :2] < hi[:2]), -1)
    libre = np.ones(local.shape[:-1], bool)
    if paredes:
        q = puntos[..., :2] @ rot_z(-giro_hab)[:2, :2].T
        for pared in paredes:
            libre &= ~((np.abs(q[..., pared[0]] - pared[1]) < 0.04) & cerca_de_pared(pared, q, 0.05))
    dentro = en_planta & libre & (local[..., 2] > lo[2]) & (local[..., 2] < hi[2])
    encima = en_planta & libre & (local[..., 2] > mueble.techo(p) + margen_arriba) & (local[..., 2] < mueble.techo(p) + 0.9)
    return dentro & ~encima, encima


def completa(mueble, p, vista, suelo):
    """Fracción de las esquinas de la caja envolvente del modelo que caen dentro de la foto."""
    x, y, giro = mueble.pose(p)
    cajas = mueble.cajas(p)
    lo, hi = np.min([c[0] for c in cajas], 0), np.max([c[1] for c in cajas], 0)
    esquinas = np.array([[a, b, c] for a in (lo[0], hi[0]) for b in (lo[1], hi[1]) for c in (lo[2], hi[2])])
    mundo = esquinas @ rot_z(giro).T + np.array([x, y, suelo])
    # Del mundo de Blender a la cámara: resolver origen + t·dir es costoso; se usa la matriz inversa de la vista.
    rel = mundo - vista.origen
    Minv = np.linalg.inv(np.stack([vista.dir_x, vista.dir_y, vista.dir_z], 1))
    cam = rel @ Minv.T
    if np.any(cam[:, 2] <= 0.05):
        return 0.0
    u = cam[:, 0] / cam[:, 2] * vista.K[0, 0] + vista.K[0, 2]
    v = cam[:, 1] / cam[:, 2] * vista.K[1, 1] + vista.K[1, 2]
    return float(np.mean((u >= 0) & (u < vista.ancho) & (v >= 0) & (v < vista.alto)))


def fuera_de_cajas(mueble, p, puntos, suelo):
    """Distancia de cada punto (mundo de Blender) a la unión de cajas del modelo (0 si está dentro)."""
    if isinstance(mueble, MallaRigida):   # distancia al vértice más cercano del modelo
        return cKDTree(mueble.puntos_mundo(p) + np.array([0, 0, suelo])).query(puntos)[0]
    x, y, giro = mueble.pose(p)
    local = (puntos - np.array([x, y, suelo])) @ rot_z(-giro).T
    d = np.full(len(puntos), np.inf)
    for lo, hi in mueble.cajas(p):
        d = np.minimum(d, np.linalg.norm(np.maximum(np.maximum(lo - local, local - hi), 0), axis=1))
    return d


def muestras_superficie(mueble, p, suelo, paso=0.04):
    """Puntos sobre las caras exteriores de las cajas del modelo (mundo de Blender), sin las caras de abajo."""
    if isinstance(mueble, MallaRigida):
        return mueble.puntos_mundo(p)[::3] + np.array([0, 0, suelo])
    x, y, giro = mueble.pose(p)
    cajas = mueble.cajas(p)
    puntos = []
    for lo, hi in cajas:
        for eje in range(3):
            for lado in ((hi[eje],) if eje == 2 else (lo[eje], hi[eje])):
                otros = [k for k in range(3) if k != eje]
                # Sin los bordes exactos: ahí las cámaras pasan rozando y la prueba de espacio libre falla.
                a = np.arange(lo[otros[0]] + paso / 2, hi[otros[0]], paso)
                b = np.arange(lo[otros[1]] + paso / 2, hi[otros[1]], paso)
                A, B = np.meshgrid(a, b)
                q = np.zeros((A.size, 3))
                q[:, otros[0]], q[:, otros[1]], q[:, eje] = A.ravel(), B.ravel(), lado
                puntos.append(q)
    q = np.concatenate(puntos)
    # Fuera las caras interiores (dentro de otra caja) y las que tocan el suelo.
    interior = np.zeros(len(q), bool)
    for lo, hi in cajas:
        interior |= np.all((q > lo + 0.005) & (q < hi - 0.005), 1)
    q = q[~interior & (q[:, 2] > 0.02)]
    return q @ rot_z(giro).T + np.array([x, y, suelo])


class EspacioLibre:
    """Prueba de espacio libre con todas las cámaras: un punto del modelo es imposible si alguna cámara vio a través
    de él (la escena en ese píxel está más lejos). Lo que ninguna cámara vio no cuenta ni a favor ni en contra."""

    def __init__(self, vistas):
        self.origen = np.stack([v.origen for v in vistas])                       # (V, 3)
        self.ejes = np.stack([np.stack([v.dir_x, v.dir_y, v.dir_z], 1) for v in vistas])  # (V, 3, 3) columnas
        self.K = np.stack([v.K for v in vistas])
        self.prof = np.stack([v.prof for v in vistas])                           # (V, H, W)
        self.H, self.W = self.prof.shape[1:]

    def violaciones(self, puntos, holgura=0.05):
        rel = puntos[None] - self.origen[:, None]                                # (V, N, 3)
        cam = np.einsum("vni,vij->vnj", rel, self.ejes)
        z = cam[..., 2]
        with np.errstate(divide="ignore", invalid="ignore"):
            u = (cam[..., 0] / z * self.K[:, None, 0, 0] + self.K[:, None, 0, 2]).astype(int)
            w = (cam[..., 1] / z * self.K[:, None, 1, 1] + self.K[:, None, 1, 2]).astype(int)
        ok = (z > 0.1) & (u >= 0) & (u < self.W) & (w >= 0) & (w < self.H)
        vv, nn = np.nonzero(ok)
        d = self.prof[vv, w[vv, nn], u[vv, nn]]
        visto = d > 0
        malo = visto & (d > z[vv, nn] + holgura)
        # Por punto: fracción de las cámaras que lo vieron que lo contradicen.
        vistos = np.bincount(nn[visto], minlength=len(puntos))
        malos = np.bincount(nn[malo], minlength=len(puntos))
        con_datos = vistos > 0
        return float((malos[con_datos] / vistos[con_datos]).mean()) if con_datos.any() else 0.0


def cara_visible(vista, mueble, p, caja, suelo):
    """Píxeles donde el rayo entra en `caja` (local) antes de chocar con la escena real (no tapados por otra cosa)."""
    x, y, giro = mueble.pose(p)
    R = rot_z(-giro)
    o = R @ (vista.origen - np.array([x, y, suelo]))
    d = vista.dir @ R.T
    lo, hi = caja
    with np.errstate(divide="ignore", invalid="ignore"):
        inv = 1.0 / d
        t1, t2 = (lo - o) * inv, (hi - o) * inv
        tmin = np.nanmax(np.minimum(t1, t2), -1)
        tmax = np.nanmin(np.maximum(t1, t2), -1)
    entra = tmax >= np.maximum(tmin, 0.05)
    # Lo que hay delante (una silla, cables) no es ni el mueble ni el hueco.
    return entra & ((vista.prof == 0) | (tmin < vista.prof + 0.05))


def luminancia(vista):
    if not hasattr(vista, "_lum"):
        img = vista.img.astype(np.float32).mean(-1)
        f = img.shape[0] // vista.alto
        vista._lum = img[:vista.alto * f, :vista.ancho * f].reshape(vista.alto, f, vista.ancho, f).mean((1, 3))
    return vista._lum


def prueba_aspecto(mueble, p, vistas, suelo):
    """Diferencia media de brillo (0-255) entre lo que debe verse como mueble y lo que debe verse como hueco."""
    if not hasattr(mueble, "partes_aspecto"):
        return 0.0
    solido, hueco = mueble.partes_aspecto(p)
    difs = []
    for v in vistas:
        a, b = cara_visible(v, mueble, p, solido, suelo), cara_visible(v, mueble, p, hueco, suelo)
        a, b = a & ~b, b & ~a
        if a.sum() >= 30 and b.sum() >= 30:
            lum = luminancia(v)
            difs.append(lum[a].mean() - lum[b].mean())
    return float(np.mean(difs)) if difs else 0.0


def iou(a, b):
    u = (a | b).sum()
    return (a & b).sum() / u if u else 0.0


# ----------------------------------------------------------------------------------------------- medidas iniciales

def altura_suelo(origen: Path) -> float:
    """Altura del suelo (eje Z de Blender): la superficie horizontal más poblada de la parte baja del LiDAR."""
    xyz, _ = s2d.leer_ply(origen / "malla_color.ply")
    z = a_blender(xyz)[:, 2]
    bajo = z[z < np.percentile(z, 15)]
    h, e = np.histogram(bajo, bins=np.arange(bajo.min(), bajo.max() + 0.01, 0.01))
    return float(e[np.argmax(h)] + 0.005)


def medidas_lidar(origen: Path, suelo: float):
    """Giro de la habitación y rectángulos de la cama (colchón) y del tablero del escritorio a partir del LiDAR."""
    xyz, _ = s2d.leer_ply(origen / "malla_color.ply")
    crudo = (origen / "malla_color.ply").read_bytes()
    fin = crudo.index(b"\n", crudo.index(b"end_header")) + 1
    dt = np.dtype([("x", "f4"), ("y", "f4"), ("z", "f4"), ("nx", "f4"), ("ny", "f4"), ("nz", "f4"),
                   ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.frombuffer(crudo[fin:fin + len(xyz) * dt.itemsize], dt)
    P = a_blender(xyz)
    N = np.stack([v["nx"], -v["nz"], v["ny"]], 1)
    h = (np.abs(N[:, 2]) < 0.2) & (P[:, 2] - suelo > 0.7) & (P[:, 2] - suelo < 2.2)
    ang = np.degrees(np.arctan2(N[h, 1], N[h, 0])) % 90
    hist, e = np.histogram(ang, bins=180, range=(0, 90))
    giro = math.radians(e[np.argmax(hist)] + 0.25)
    Q = P[:, :2] @ rot_z(-giro)[:2, :2].T
    alto = P[:, 2] - suelo

    def caja(h0, h1, region, p=3):
        m = (N[:, 2] > 0.85) & (alto > h0) & (alto < h1)
        m &= (Q[:, 0] > region[0]) & (Q[:, 0] < region[1]) & (Q[:, 1] > region[2]) & (Q[:, 1] < region[3])
        lo, hi = np.percentile(Q[m], p, 0), np.percentile(Q[m], 100 - p, 0)
        centro = rot_z(giro)[:2, :2] @ ((lo + hi) / 2)
        return centro, hi - lo, float(np.median(alto[m]))

    # Paredes: picos de posición de las superficies verticales a lo largo de cada eje de la habitación.
    paredes = []
    vert = (np.abs(N[:, 2]) < 0.2) & (alto > 0.3)
    Nq = N[:, :2] @ rot_z(-giro)[:2, :2].T
    for eje in (0, 1):
        sel = vert & (np.abs(Nq[:, eje]) > 0.9)
        h, e = np.histogram(Q[sel, eje], bins=np.arange(Q[:, eje].min() - 0.02, Q[:, eje].max() + 0.02, 0.01))
        for k in np.argsort(h)[::-1]:
            if h[k] < 4000:
                break
            pos = e[k] + 0.005
            if all(abs(pos - p[1]) > 0.15 for p in paredes if p[0] == eje):
                # Tramo que ocupa la pared a lo largo del otro eje (hay tabiques cortos, no son infinitas).
                en = sel & (np.abs(Q[:, eje] - pos) < 0.03) & (alto > 1.9)   # por encima de los muebles
                if en.sum() < 200:
                    continue
                lo, hi = np.percentile(Q[en, 1 - eje], [1, 99])
                paredes.append((eje, float(pos), float(lo), float(hi)))
    return giro, caja, paredes


# ----------------------------------------------------------------------------------------------- segmentación

def segmentar(P, caras, giro, paredes, suelo, corte_suelo=0.10, corte_pared=0.08, minimo=3000):
    """Piezas conexas de la malla sin suelo, paredes ni techo: [{indices, lo, hi}] en ejes de la habitación."""
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    Q = P[:, :2] @ rot_z(-giro)[:2, :2].T
    alto = P[:, 2] - suelo
    estructura = (alto < corte_suelo) | (alto > np.percentile(alto, 99.5) - 0.08)
    for pared in paredes:
        estructura |= (np.abs(Q[:, pared[0]] - pared[1]) < corte_pared) & cerca_de_pared(pared, Q, 0.05)
    f = caras[~estructura[caras].any(1)]
    n = len(P)
    a, b = np.r_[f[:, 0], f[:, 1]], np.r_[f[:, 1], f[:, 2]]
    _, etiqueta = connected_components(coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n)), directed=False)
    usados = np.unique(f)
    tam = np.bincount(etiqueta[usados], minlength=etiqueta.max() + 1)
    piezas = []
    for k in np.argsort(tam)[::-1]:
        if tam[k] < minimo:
            break
        idx = usados[etiqueta[usados] == k]
        piezas.append({"indices": idx, "lo": np.percentile(Q[idx], 1, 0), "hi": np.percentile(Q[idx], 99, 0)})
    return piezas


def clasificar(piezas, P, caras, suelo):
    """Reconoce muebles por su forma: altura de la superficie horizontal principal, planta y altura total.
    Devuelve [(nombre, tipo, pieza)] con pieza["techo"] = altura de esa superficie y pieza["girar"] si el lado
    largo va a lo largo del eje y de la habitación."""
    fn = np.cross(P[caras[:, 1]] - P[caras[:, 0]], P[caras[:, 2]] - P[caras[:, 0]])
    n = np.zeros_like(P)
    for k in range(3):
        np.add.at(n, caras[:, k], fn)
    n /= np.linalg.norm(n, axis=1, keepdims=True) + 1e-12
    alto = P[:, 2] - suelo
    reconocidos, cuenta = [], {}
    for pieza in piezas:
        i = pieza["indices"]
        horiz = i[np.abs(n[i, 2]) > 0.9]
        if len(horiz) < 200:
            continue
        h, e = np.histogram(alto[horiz], bins=np.arange(0.1, 2.5, 0.02))
        techo = float(e[np.argmax(h)] + 0.01)
        frac = len(horiz) / len(i)
        L, W = pieza["hi"] - pieza["lo"]
        largo, ancho = max(L, W), min(L, W)
        p98 = float(np.percentile(alto[i], 98))
        tipo = None
        if 0.35 <= techo <= 0.80 and largo >= 1.5 and ancho >= 0.8 and p98 < 1.2 and frac > 0.2:
            tipo = "cama"
        elif 0.65 <= techo <= 0.85 and 0.7 <= largo <= 2.4 and 0.35 <= ancho <= 1.0:
            tipo = "escritorio"
        elif 0.40 <= techo <= 0.62 and 0.45 <= largo <= 0.85 and 0.45 <= ancho <= 0.85 and 0.85 <= p98 <= 1.5:
            tipo = "silla"
        if tipo:
            cuenta[tipo] = cuenta.get(tipo, 0) + 1
            nombre = tipo if cuenta[tipo] == 1 else f"{tipo}_{cuenta[tipo]}"
            reconocidos.append((nombre, tipo, dict(pieza, techo=techo, girar=bool(W > L))))
    return reconocidos


def zona_pieza(pieza, puntos, giro, paredes, suelo, margen=0.05):
    """Como `zona`, pero con la caja fija de la pieza segmentada (no depende del modelo que se está ajustando)."""
    q = puntos[..., :2] @ rot_z(-giro)[:2, :2].T
    alto = puntos[..., 2] - suelo
    en_planta = np.all((q > pieza["lo"] - margen) & (q < pieza["hi"] + margen), -1)
    for pared in paredes:
        en_planta &= ~((np.abs(q[..., pared[0]] - pared[1]) < 0.04) & cerca_de_pared(pared, q, 0.05))
    techo = pieza["techo"]
    dentro = en_planta & (alto > 0.03) & (alto < techo + 0.02)
    encima = en_planta & (alto >= techo + 0.02) & (alto < techo + 0.9)
    return dentro, encima


# ----------------------------------------------------------------------------------------------- escena para la app

def malla_de_cajas(cajas, colores):
    """Triángulos de una unión de cajas (locales) con un color por caja."""
    v, f, c = [], [], []
    caras = [(0, 1, 3), (0, 3, 2), (4, 6, 7), (4, 7, 5), (0, 4, 5), (0, 5, 1),
             (2, 3, 7), (2, 7, 6), (0, 2, 6), (0, 6, 4), (1, 5, 7), (1, 7, 3)]
    for (lo, hi), col in zip(cajas, colores):
        base = len(v)
        v += [(x, y, z) for x in (lo[0], hi[0]) for y in (lo[1], hi[1]) for z in (lo[2], hi[2])]
        f += [(base + a, base + b, base + d) for a, b, d in caras]
        c += [col] * 8
    return np.array(v, float), np.array(f, np.int32), np.array(c, np.uint8)


def color_en(color, alto, desde, hasta):
    sel = (alto >= desde) & (alto < hasta)
    return np.median(color[sel], 0).astype(np.uint8) if sel.sum() > 50 else np.array([150, 150, 150], np.uint8)


def exportar_escena(origen, muebles, de_pieza, suelo, salida, paredes=(), giro=0.0):
    """Escena para la app (mundo ARKit, Y arriba): la malla escaneada sin los muebles reconocidos + sus modelos
    limpios con el color real de cada parte. Lo que hay encima de los muebles (monitor, PC…) sigue escaneado."""
    from splat2d_local import guardar_malla_ply

    base = origen / "splat2d" / "malla_movil.ply"
    if not base.exists():
        base = origen / "splat2d" / "malla_2dgs_plana.ply"
    va, ca, fa = leer(base)
    vb = a_blender(va)
    centros = vb[fa].mean(1)
    quitar = np.zeros(len(fa), bool)
    piezas_v, piezas_f, piezas_c = [], [], []
    v_plana, c_plana, _ = leer(origen / "splat2d" / "malla_2dgs_plana.ply")
    pb = a_blender(v_plana)
    for nombre, mueble in muebles.items():
        p = mueble.p
        idx = de_pieza[nombre]["indices"]
        alto, color = pb[idx, 2] - suelo, c_plana[idx]
        if isinstance(mueble, MallaRigida):
            pts = mueble.puntos_mundo(p, todos=True) + np.array([0, 0, suelo])
            quitar |= cKDTree(pts).query(centros)[0] < 0.05
            n0 = len(np.load(origen / "modelado" / f"{nombre.split('_')[0]}_ia.npz")["v"])
            caras_ia = np.load(origen / "modelado" / f"{nombre.split('_')[0]}_ia.npz")["f"]
            v_obj = pts[:n0]
            col = np.tile(color_en(color, alto, 0.1, 1.6), (len(v_obj), 1))
            piezas_v.append(v_obj), piezas_f.append(caras_ia), piezas_c.append(col)
            continue
        x, y, giro = mueble.pose(p)
        cajas = mueble.cajas(p)
        local = (centros - np.array([x, y, suelo])) @ rot_z(-giro).T
        for lo, hi in cajas:
            quitar |= np.all((local > lo - 0.03) & (local < hi + 0.03), 1)
        if isinstance(mueble, Cama):
            H = p[5]
            colores = [color_en(color, alto, H - 0.12, H + 0.02), color_en(color, alto, H + 0.02, H + 0.2),
                       color_en(color, alto, H + 0.02, H + 0.2)]
            # Base y colchón por separado: la base toma el color de la parte baja.
            lo, hi = cajas[0]
            b = min(0.40, H - 0.18)
            cajas = [(lo, np.array([hi[0], hi[1], b])), (np.array([lo[0], lo[1], b]), hi)] + cajas[1:]
            colores = [color_en(color, alto, 0.12, b - 0.03)] + colores
        else:
            colores = [color_en(color, alto, 0.12, p[5] + 0.02)] * len(cajas)
        v_obj, f_obj, c_obj = malla_de_cajas(cajas, colores)
        piezas_v.append(v_obj @ rot_z(giro).T + np.array([x, y, suelo])), piezas_f.append(f_obj), piezas_c.append(c_obj)
    fa = fa[~quitar]
    # Cascarón: paredes y suelo planos 2 cm por detrás del escaneo, con su color real, para rellenar los huecos de
    # las zonas lisas que ninguna cámara cubrió. Cada plano tiene una sola cara, mirando hacia dentro: desde fuera de
    # la habitación no tapa nada (vista de casa de muñecas). Solo superficies que van del suelo al techo.
    if paredes:
        q_todos = pb[:, :2] @ rot_z(-giro)[:2, :2].T
        alto_todos = pb[:, 2] - suelo
        centro_hab = np.median(q_todos, 0)
        # Para el color de las paredes no valen los puntos de los muebles pegados a ellas (sombras del escritorio…).
        de_mueble = np.zeros(len(pb), bool)
        for pieza in de_pieza.values():
            de_mueble[pieza["indices"]] = True
        techo_alto = float(np.percentile(alto_todos, 99.5)) - 0.05
        v_c, f_c, c_c = [], [], []

        def plano(esquinas, color, normal_deseada):
            a, b, c_, d = esquinas
            if np.dot(np.cross(b - a, c_ - a), normal_deseada) < 0:
                a, b, c_, d = a, d, c_, b
            base = len(v_c)
            v_c.extend([a, b, c_, d])
            f_c.extend([(base, base + 1, base + 2), (base, base + 2, base + 3)])
            c_c.extend([color] * 4)

        for eje, pos, lo_t, hi_t in paredes:
            cerca = (np.abs(q_todos[:, eje] - pos) < 0.03) & (q_todos[:, 1 - eje] > lo_t) & (q_todos[:, 1 - eje] < hi_t)
            cerca &= ~de_mueble
            if hi_t - lo_t < 1.5 or cerca.sum() < 200:   # los tabiques cortos tienen dos lados «interiores»
                continue
            if np.percentile(alto_todos[cerca], 5) > 0.8 or np.percentile(alto_todos[cerca], 95) < techo_alto - 0.5:
                continue   # no va del suelo al techo: un mueble alto, no una pared
            hacia_dentro = np.sign(centro_hab[eje] - pos)
            x = pos - hacia_dentro * 0.02
            # Rejilla de 10 cm con el color del escaneo cercano en cada punto (rosa, blanco, sombras…).
            arbol = cKDTree(np.c_[q_todos[cerca, 1 - eje], alto_todos[cerca]])
            ts = np.linspace(lo_t, hi_t, max(2, int((hi_t - lo_t) / 0.1) + 1))
            zs = np.linspace(0.0, techo_alto, max(2, int(techo_alto / 0.1) + 1))
            T_, Z_ = np.meshgrid(ts, zs)
            _, vecinos = arbol.query(np.c_[T_.ravel(), Z_.ravel()], k=12)
            colores_rej = c_plana[cerca][vecinos].mean(1).astype(np.uint8)
            base = len(v_c)
            for t, z, col in zip(T_.ravel(), Z_.ravel(), colores_rej):
                q = np.zeros(2)
                q[eje], q[1 - eje] = x, t
                v_c.append(np.array([*q, z]))
                c_c.append(col)
            nx_ = len(ts)
            n = np.zeros(3)
            n[eje] = hacia_dentro
            for j in range(len(zs) - 1):
                for i in range(nx_ - 1):
                    a0, a1 = base + j * nx_ + i, base + j * nx_ + i + 1
                    b0, b1 = a0 + nx_, a1 + nx_
                    tri = [(a0, a1, b1), (a0, b1, b0)]
                    if np.dot(np.cross(v_c[a1] - v_c[a0], v_c[b1] - v_c[a0]), n) < 0:
                        tri = [(a0, b1, a1), (a0, b0, b1)]
                    f_c.extend(tri)
        lo_s, hi_s = np.percentile(q_todos, 1, 0), np.percentile(q_todos, 99, 0)
        suelo_sel = np.abs(alto_todos) < 0.02
        col_s = np.median(c_plana[suelo_sel], 0).astype(np.uint8) if suelo_sel.sum() > 100 else np.array([190, 180, 170], np.uint8)
        plano([np.array([lo_s[0], lo_s[1], -0.02]), np.array([hi_s[0], lo_s[1], -0.02]),
               np.array([hi_s[0], hi_s[1], -0.02]), np.array([lo_s[0], hi_s[1], -0.02])], col_s, np.array([0, 0, 1.0]))
        V_c = np.array(v_c)
        V_c = np.c_[V_c[:, :2] @ rot_z(giro)[:2, :2].T, V_c[:, 2] + suelo]
        piezas_v.append(V_c), piezas_f.append(np.array(f_c, np.int32)), piezas_c.append(np.array(c_c, np.uint8))
    todos_v, todos_f, todos_c = [vb], [fa], [ca]
    desplazamiento = len(vb)
    for v_obj, f_obj, c_obj in zip(piezas_v, piezas_f, piezas_c):
        todos_v.append(v_obj), todos_f.append(f_obj + desplazamiento), todos_c.append(c_obj)
        desplazamiento += len(v_obj)
    V = np.concatenate(todos_v)
    V = np.stack([V[:, 0], V[:, 2], -V[:, 1]], 1)       # Blender → ARKit (Y arriba), como el resto de mallas
    guardar_malla_ply(salida, V.astype(np.float32), np.concatenate(todos_c).astype(np.uint8),
                      np.concatenate(todos_f).astype(np.int32))
    print(f"Escena modelada: {quitar.sum()} triángulos escaneados sustituidos por {len(muebles)} modelos → {salida}")


# ----------------------------------------------------------------------------------------------- principal

def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    p.add_argument("--vistas", type=int, default=10)
    p.add_argument("--solo", help="modelar solo este mueble")
    a = p.parse_args()

    origen = Path(a.carpeta).resolve()
    trabajo = s2d.HERRAMIENTAS / "trabajos" / origen.name
    destino = origen / "modelado"
    destino.mkdir(exist_ok=True)
    T = np.array(json.loads((origen / "fotogrametria" / "colmap_a_lidar.json").read_text())["matriz"])
    suelo = altura_suelo(origen)
    giro, caja, paredes = medidas_lidar(origen, suelo)
    print(f"Suelo a {suelo:.3f} m")
    print(f"Giro de la habitación {math.degrees(giro):.1f}°; paredes en {[(e, round(p, 2)) for e, p, *_ in paredes]}")

    # Segmentación: sin suelo, paredes ni techo, cada mueble es una pieza conexa de la malla 2DGS (con lo que
    # tenga encima). La pieza da sus puntos completos —también lo que ninguna foto ve— y su caja inicial.
    v_malla, _, f_malla = leer(origen / "splat2d" / "malla_2dgs_plana.ply")
    malla = a_blender(v_malla)
    piezas = segmentar(malla, f_malla, giro, paredes, suelo)
    print(f"{len(piezas)} piezas: " + ", ".join(f"{(p_['hi'] - p_['lo'])[0]:.2f}×{(p_['hi'] - p_['lo'])[1]:.2f}"
                                                for p_ in piezas[:8]))
    # Reconocimiento: cada pieza se identifica por su forma (cama, escritorio, silla); el resto queda escaneado.
    muebles, de_pieza = {}, {}
    for nombre, tipo, pieza in clasificar(piezas, malla, f_malla, suelo):
        lo, hi = pieza["lo"], pieza["hi"]
        c = rot_z(giro)[:2, :2] @ ((lo + hi) / 2)
        L, W = hi - lo
        g0 = giro + (math.pi / 2 if pieza["girar"] else 0.0)
        if pieza["girar"]:
            L, W = W, L
        if tipo == "cama":
            muebles[nombre] = Cama([c[0], c[1], g0, L, W, pieza["techo"]])
        elif tipo == "escritorio":
            muebles[nombre] = Escritorio([c[0], c[1], g0, L, W, pieza["techo"], 0.45])
        elif tipo == "silla":
            # Modelo detallado (generado por IA) si lo hay: <destino>/silla_ia.npz (vértices, caras y pose).
            archivo = destino / f"{tipo}_ia.npz"
            if not archivo.exists():
                print(f"{nombre}: reconocida, pero sin modelo de IA ({archivo.name}); se queda escaneada")
                continue
            datos_ia = np.load(archivo)
            # Puede quedar partida en varias piezas (asiento, brazos): se unen las bajas cercanas.
            centro_q = (lo + hi) / 2
            cerca = [p_ for p_ in piezas if np.percentile(malla[p_["indices"], 2] - suelo, 99) < 1.6
                     and np.linalg.norm((p_["lo"] + p_["hi"]) / 2 - centro_q) < 0.6]
            pieza = {"indices": np.concatenate([p_["indices"] for p_ in cerca]),
                     "lo": np.min([p_["lo"] for p_ in cerca], 0), "hi": np.max([p_["hi"] for p_ in cerca], 0),
                     "techo": 3.0}
            c = rot_z(giro)[:2, :2] @ ((pieza["lo"] + pieza["hi"]) / 2)
            muebles[nombre] = MallaRigida([c[0], c[1], float(datos_ia["pose"][2]), 1.0], datos_ia["v"], datos_ia["f"])
        de_pieza[nombre] = pieza
        print(f"{nombre}: pieza de {L:.2f} × {W:.2f} m, superficie a {pieza['techo']:.2f} m")

    vistas_todas = s2d.cargar_vistas(trabajo, T, 1280)
    datos = torch.load(origen / "splat2d" / "splats.pt")
    validas = ~torch.cat([torch.isnan(v.reshape(len(v), -1)) for v in datos.values()], 1).any(1)
    dev = "cuda"
    splats = {k: v[validas].to(dev) for k, v in datos.items()}
    grado = int(math.isqrt(splats["shN"].shape[1] + 1)) - 1
    from gsplat import rasterization_2dgs

    def render(v):
        K = v["K"].copy()
        K[:2] *= RES
        w, hh = int(v["ancho"] * RES), int(v["alto"] * RES)
        with torch.no_grad():
            _, alfa, _, _, _, med, _ = rasterization_2dgs(
                means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
                opacities=torch.sigmoid(splats["opacities"]), colors=torch.cat([splats["sh0"], splats["shN"]], 1),
                viewmats=torch.from_numpy(v["w2c"]).to(dev)[None], Ks=torch.from_numpy(K).to(dev)[None],
                width=w, height=hh, sh_degree=grado, render_mode="RGB+ED", packed=False, near_plane=0.05,
                far_plane=20.0)
        return med[0, ..., 0].cpu().numpy(), alfa[0, ..., 0].cpu().numpy()

    print("Profundidad de la escena desde cada cámara…")
    todas = [Vista(v, *render(v)) for v in vistas_todas]
    espacio_libre = EspacioLibre(todas)

    def referencias(mueble, p, vistas):
        pieza = de_pieza[nombre]
        salida = []
        for v in vistas:
            obj, neutro = zona_pieza(pieza, v.puntos, giro, paredes, suelo)
            salida.append((obj & (v.prof > 0), neutro))
        return salida

    informe = {}
    for nombre, mueble in muebles.items():
        if a.solo and nombre != a.solo:
            continue
        t0 = time.time()
        # Vistas que muestran el mueble entero y grande, repartidas (no casi la misma toma varias veces).
        puntuadas = []
        for v in todas:
            completo = completa(mueble, mueble.p, v, suelo)
            if completo < 0.25:
                continue
            area = referencias(mueble, mueble.p, [v])[0][0].mean()
            puntuadas.append((area * completo, v))
        puntuadas.sort(key=lambda c: -c[0])
        vistas = []
        for f, v in puntuadas:
            if f < 0.02 or len(vistas) == a.vistas:
                break
            if all(np.linalg.norm(v.origen - w.origen) > 0.15 for w in vistas):
                vistas.append(v)
        refs = referencias(mueble, mueble.p, vistas)

        # Geometría 3D (malla 2DGS guiada por el LiDAR): completa lo que ninguna foto ve (p. ej. bajo las almohadas).
        pieza = de_pieza[nombre]
        rng = np.random.default_rng(0)
        todos = malla[pieza["indices"]]
        bajo = todos[todos[:, 2] - suelo < pieza["techo"] + 0.02]     # lo de encima no es el mueble
        nube = bajo[rng.choice(len(bajo), min(15_000, len(bajo)), replace=False)]
        arbol = cKDTree(todos[rng.choice(len(todos), min(30_000, len(todos)), replace=False)])

        def puntos_3d(p):
            return nube

        def coste_3d(p):
            fuera = np.minimum(fuera_de_cajas(mueble, p, nube, suelo), 0.2).mean()
            sobra = np.minimum(arbol.query(muestras_superficie(mueble, p, suelo))[0], 0.2).mean()
            return (fuera + sobra) / 0.05   # 5 cm de error medio ≈ 1 punto de IoU

        def desglose(p):
            fotos = 1 - np.mean([iou(v.silueta_modelo(mueble, p, suelo)[0] & ~n, r) for v, (r, n) in zip(vistas, refs)])
            fuera = np.minimum(fuera_de_cajas(mueble, p, nube, suelo), 0.2).mean()
            libre = espacio_libre.violaciones(muestras_superficie(mueble, p, suelo, paso=0.06))
            return fotos, fuera, libre

        centro_q = rot_z(-giro)[:2, :2] @ mueble.p[:2]

        def atraviesa_paredes(p):
            """Metros que las esquinas del modelo se meten en las paredes (un mueble no las atraviesa)."""
            x_, y_, g_ = mueble.pose(p)
            cajas = mueble.cajas(p)
            lo, hi = np.min([c[0] for c in cajas], 0), np.max([c[1] for c in cajas], 0)
            esquinas = np.array([[a_, b_] for a_ in (lo[0], hi[0]) for b_ in (lo[1], hi[1])])
            q = (esquinas @ rot_z(g_)[:2, :2].T + np.array([x_, y_])) @ rot_z(-giro)[:2, :2].T
            total = 0.0
            for pared in paredes:
                eje, pos = pared[0], pared[1]
                lado = np.sign(pos - centro_q[eje])       # la pared está a este lado del mueble
                if abs(pos - centro_q[eje]) > 1.5 or not cerca_de_pared(pared, centro_q, 0.1):
                    continue
                total += (np.maximum(lado * (q[:, eje] - pos) + 0.01, 0) * cerca_de_pared(pared, q)).sum()
            return total

        def coste(p):
            fotos, fuera, libre = desglose(p)
            # 1 punto de IoU ≈ 5 cm de puntos fuera ≈ 10 % de la superficie del modelo vista a través ≈ 2 cm en la pared.
            return fotos + 0.1 * fuera / 0.05 + 0.1 * libre / 0.10 + 0.1 * atraviesa_paredes(p) / 0.02

        # Hacia dónde mira: las siluetas no lo distinguen, pero los muebles apoyan su espalda (escritorio) o su
        # cabecera (cama) en la pared más cercana por ese lado.
        mueble.p[2] = orientar(mueble, mueble.p, paredes, giro)
        if isinstance(mueble, MallaRigida):
            # Escala por la altura real de la pieza y giro por búsqueda gruesa (cada 30°): el modelo de IA viene en
            # cualquier orientación y tamaño.
            alto_real = np.percentile(malla[pieza["indices"], 2] - suelo, 99)
            mueble.p[3] = alto_real / np.ptp(mueble.v[:, 2])
            candidatos = []
            for g in np.radians(np.arange(0, 360, 30)):
                prueba = mueble.p.copy()
                prueba[2] = g
                candidatos.append((coste(prueba), g))
            mueble.p[2] = min(candidatos)[1]
            print(f"   escala inicial {mueble.p[3]:.2f}, giro inicial {math.degrees(mueble.p[2]):.0f}°")
        # En qué extremo está el cajonero: el que mejor explique fotos, puntos y espacio libre.
        mejor = None
        for lado in ((1, -1) if isinstance(mueble, Escritorio) else (1,)):
            mueble.lado = lado
            valor = 1 - coste(mueble.p)
            if mejor is None or valor > mejor[0]:
                mejor = (valor, lado)
        inicial, mueble.lado = mejor
        print(f"{nombre}: mira hacia {math.degrees(mueble.p[2] - giro) % 360:.0f}° respecto a la habitación, lado {mueble.lado}")
        pasos = np.array([0.05, 0.05, math.radians(3)] + [0.05] * (len(mueble.p) - 3))
        p_ini = mueble.p.copy()
        # Rondas: ajustar, y recalcular la referencia alrededor del modelo ajustado (así puede crecer hasta el
        # borde real del mueble), hasta que deje de moverse.
        for ronda in range(8):
            simplex = np.vstack([mueble.p] + [mueble.p + np.eye(len(mueble.p))[i] * pasos[i]
                                              for i in range(len(mueble.p))])
            res = minimize(coste, mueble.p, method="Nelder-Mead",
                           options={"initial_simplex": simplex, "maxiter": 400, "xatol": 1e-3, "fatol": 1e-4})
            cambio = np.abs(res.x - mueble.p).max()
            mueble.p = res.x
            refs[:] = referencias(mueble, mueble.p, vistas)
            print(f"   ronda {ronda + 1}: IoU {1 - coste(mueble.p):.3f}, cambio máximo {cambio:.3f}")
            if cambio < 0.01:
                break
        # Aspecto: la geometría no distingue un cajonero de un hueco lleno de cables; la foto sí (claro frente a
        # oscuro). Se prueba con todas las cámaras y, si el otro lado se ve mucho mejor, se cambia y se reajusta.
        if hasattr(mueble, "partes_aspecto"):
            puntos_lado = {}
            for lado in (1, -1):
                mueble.lado = lado
                puntos_lado[lado] = prueba_aspecto(mueble, mueble.p, todas, suelo)
            mueble.lado = max(puntos_lado, key=puntos_lado.get)
            print(f"   aspecto (claro cajonero − oscuro hueco): lado +1 {puntos_lado[1]:.1f}, lado −1 {puntos_lado[-1]:.1f}"
                  f" → lado {mueble.lado:+d}")
            res = minimize(coste, mueble.p, method="Nelder-Mead", options={"maxiter": 200, "xatol": 1e-3, "fatol": 1e-4})
            mueble.p = res.x
        final = 1 - coste(mueble.p)
        # Diagnóstico: superficie vista a través, caja por caja y cara por cara (modelo inicial).
        x_, y_, g_ = mueble.pose(p_ini)
        for k, (lo, hi) in enumerate(mueble.cajas(p_ini)):
            caras = {}
            for eje, nom in ((0, "x"), (1, "y"), (2, "z")):
                for lado, val in (("-", lo[eje]), ("+", hi[eje])):
                    if eje == 2 and lado == "-":
                        continue
                    otros = [j for j in range(3) if j != eje]
                    A, B = np.meshgrid(np.linspace(lo[otros[0]], hi[otros[0]], 12), np.linspace(lo[otros[1]], hi[otros[1]], 12))
                    q = np.zeros((A.size, 3))
                    q[:, otros[0]], q[:, otros[1]], q[:, eje] = A.ravel(), B.ravel(), val
                    q = q[q[:, 2] > 0.02] @ rot_z(g_).T + np.array([x_, y_, suelo])
                    caras[nom + lado] = round(100 * espacio_libre.violaciones(q), 1)
            print(f"   caja {k}: vista a través (%) {caras}")
        for etiqueta, pp in (("inicial", p_ini), ("final", mueble.p)):
            fo, fu, so = desglose(pp)
            print(f"   {etiqueta}: IoU fotos {1 - fo:.3f} · puntos fuera {fu * 100:.1f} cm · superficie vista a través "
                  f"{so * 100:.1f} % · dentro de paredes {atraviesa_paredes(pp) * 100:.1f} cm")
        informe[nombre] = {"tipo": type(mueble).__name__, "lado": getattr(mueble, "lado", 1),
                           "parametros": dict(zip(mueble.nombres, map(float, res.x))),
                           "iou_inicial": float(inicial), "iou_final": float(final),
                           "vistas": [v.nombre for v in vistas]}
        print(f"{nombre}: IoU {inicial:.3f} -> {final:.3f} en {len(vistas)} vistas ({time.time() - t0:.0f} s)")
        for k, n in enumerate(mueble.nombres):
            u = "°" if n == "giro" else " m"
            f = math.degrees if n == "giro" else (lambda x: x)
            print(f"   {n:15s} {f(p_ini[k]):8.3f} -> {f(res.x[k]):8.3f}{u}")

        # Superposiciones: verde = objeto real (referencia), rojo = modelo inicial / final.
        hoja = []
        for v, (r, n) in zip(vistas, refs):
            foto = Image.fromarray(v.img).resize((v.ancho * 2, v.alto * 2))
            capa = np.zeros((v.alto, v.ancho, 4), np.uint8)
            capa[r] = (0, 255, 0, 90)
            capa[n] = (0, 120, 255, 70)
            ini = v.silueta_modelo(mueble, p_ini, suelo)[0]
            fin = v.silueta_modelo(mueble, res.x, suelo)[0]
            borde = lambda m: m & ~(np.roll(m, 1, 0) & np.roll(m, -1, 0) & np.roll(m, 1, 1) & np.roll(m, -1, 1))
            capa[borde(ini)] = (255, 200, 0, 255)
            capa[borde(fin)] = (255, 0, 0, 255)
            foto.paste(Image.fromarray(capa).resize(foto.size, Image.NEAREST), (0, 0), Image.fromarray(capa).resize(foto.size, Image.NEAREST))
            ImageDraw.Draw(foto).text((6, 6), f"{v.nombre}  IoU {iou(fin & ~n, r):.2f}", fill=(255, 255, 255))
            hoja.append(foto)
        w, hh = hoja[0].size
        lienzo = Image.new("RGB", (w * 4, hh * math.ceil(len(hoja) / 4)))
        for k, im in enumerate(hoja):
            lienzo.paste(im.resize((w, hh)), ((k % 4) * w, (k // 4) * hh))
        lienzo.save(destino / f"{nombre}.jpg", quality=88)

    informe["_escena"] = {"suelo": suelo, "giro_habitacion": giro}
    (destino / "modelado.json").write_text(json.dumps(informe, indent=1, ensure_ascii=False), encoding="utf-8")
    if not a.solo and muebles:
        exportar_escena(origen, muebles, de_pieza, suelo, origen / "splat2d" / "escena_modelada.ply", paredes, giro)


if __name__ == "__main__":
    main()
