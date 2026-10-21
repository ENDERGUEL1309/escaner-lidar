r"""Ajusta suelo, techo y paredes de una malla (mundo ARKit, Y arriba) a planos exactos, tras un suavizado Taubin.

    .venv-splat\Scripts\python aplanar_malla.py malla.ply [--tolerancia 0.03]

Los planos se detectan solos: suelo y techo por la altura de las superficies horizontales; las paredes por el giro
de la habitación (normales horizontales, módulo 90°) y los picos de posición a lo largo de cada eje. Solo se mueven
los vértices que están cerca de un plano y orientados como él, así que muebles y objetos quedan intactos.
Escribe <malla>_plana.ply.
"""

from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags

from splat2d_local import guardar_malla_ply
from suavizar_malla import leer


def normales_vertice(v, f):
    fn = np.cross(v[f[:, 1]] - v[f[:, 0]], v[f[:, 2]] - v[f[:, 0]])
    n = np.zeros_like(v)
    for k in range(3):
        np.add.at(n, f[:, k], fn)
    return n / (np.linalg.norm(n, axis=1, keepdims=True) + 1e-12)


def taubin(v, f, pasadas):
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = coo_matrix((np.ones(len(i)), (i, j)), shape=(len(v), len(v))).tocsr()
    A.data[:] = 1
    L = diags(1 / np.maximum(A.sum(1).A1, 1)) @ A
    for _ in range(pasadas):
        v = v + 0.5 * (L @ v - v)
        v = v - 0.53 * (L @ v - v)
    return v


def ajustar_plano(p):
    """Plano por mínimos cuadrados: (normal unitaria, desplazamiento) con n·x = d."""
    c = p.mean(0)
    n = np.linalg.svd(p - c, full_matrices=False)[2][-1]
    return n, float(n @ c)


def picos(valores, paso=0.01, minimo=3000, separacion=0.12):
    """Posiciones con mucha superficie (histograma suavizado), separadas al menos `separacion`."""
    if len(valores) == 0:
        return []
    bordes = np.arange(valores.min() - paso, valores.max() + 2 * paso, paso)
    h, _ = np.histogram(valores, bordes)
    h = np.convolve(h, np.ones(3), "same")
    salida = []
    for k in np.argsort(h)[::-1]:
        if h[k] < minimo:
            break
        x = bordes[k] + paso / 2
        if all(abs(x - s) > separacion for s in salida):
            salida.append(x)
    return salida


def detectar_planos(v, n, tolerancia):
    planos = []
    Y = np.array([0.0, 1.0, 0.0])

    def refinar(mascara, n0, nombre, alto_minimo=0.0):
        if mascara.sum() < 500:
            return
        # Una pared tiene que cubrir buena parte de la altura (los frentes de muebles no).
        if alto_minimo and np.ptp(np.percentile(v[mascara, 1], [2, 98])) < alto_minimo:
            return
        nn, d = ajustar_plano(v[mascara])
        if nn @ n0 < 0:
            nn, d = -nn, -d
        planos.append((nombre, nn, d))

    # Suelo (normal hacia arriba, en la mitad baja) y techo (normal hacia abajo, en la mitad alta).
    medio = np.median(v[:, 1])
    arriba, abajo = n[:, 1] > 0.9, n[:, 1] < -0.9
    for nombre, sel, n0, extremo in (("suelo", arriba & (v[:, 1] < medio), Y, min),
                                     ("techo", abajo & (v[:, 1] > medio), -Y, max)):
        cand = picos(v[sel, 1])
        if cand:
            h = extremo(cand[:3])
            refinar(sel & (abs(v[:, 1] - h) < tolerancia), n0, nombre)

    # Paredes: giro de la habitación a partir de las normales horizontales.
    horiz = np.abs(n[:, 1]) < 0.25
    ang = np.degrees(np.arctan2(n[horiz, 2], n[horiz, 0])) % 90
    h, e = np.histogram(ang, bins=180, range=(0, 90))
    giro = np.radians(e[np.argmax(np.convolve(h, np.ones(3), "same"))] + 0.25)
    for k, eje in enumerate((np.array([np.cos(giro), 0, np.sin(giro)]),
                             np.array([-np.sin(giro), 0, np.cos(giro)]))):
        for signo in (1, -1):
            sel = horiz & (n @ (signo * eje) > 0.85)
            for pos in picos(v[sel] @ eje, minimo=8000):
                refinar(sel & (abs(v @ eje - pos) < tolerancia), signo * eje, f"pared_{k}{'+' if signo > 0 else '-'}",
                        alto_minimo=1.5)
    return planos, np.degrees(giro)


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("malla")
    p.add_argument("--tolerancia", type=float, default=0.03, help="distancia máxima al plano para ajustar (m)")
    p.add_argument("--pasadas", type=int, default=10, help="pasadas de suavizado Taubin previas")
    a = p.parse_args()

    ruta = Path(a.malla)
    v, c, f = leer(ruta)
    v = taubin(v, f, a.pasadas)
    n = normales_vertice(v, f)
    planos, giro = detectar_planos(v, n, a.tolerancia)
    print(f"Giro de la habitación: {giro:.1f}°; {len(planos)} planos")
    movidos = np.zeros(len(v), bool)
    for nombre, nn, d in planos:
        dist = v @ nn - d
        sel = (np.abs(dist) < a.tolerancia) & (n @ nn > 0.7) & ~movidos
        v[sel] -= dist[sel, None] * nn
        movidos |= sel
        print(f"  {nombre:9s} normal {np.round(nn, 3)}  d={d:+.3f}  vértices ajustados {sel.sum()}")
    salida = ruta.with_name(ruta.stem + "_plana.ply")
    guardar_malla_ply(salida, v.astype(np.float32), c, f)
    print(f"{movidos.sum()} de {len(v)} vértices sobre planos -> {salida}")


if __name__ == "__main__":
    main()
