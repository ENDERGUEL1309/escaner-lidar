r"""Suavizado Taubin (sin encoger) de una malla PLY con color por vértice: quita el ondulado de las paredes lisas.

    .venv-splat\Scripts\python suavizar_malla.py malla.ply [--pasadas 10]
"""

import argparse
from pathlib import Path

import numpy as np
from scipy.sparse import coo_matrix, diags

from splat2d_local import guardar_malla_ply


def leer(ruta: Path):
    crudo = ruta.read_bytes()
    fin = crudo.index(b"end_header\n") + len(b"end_header\n")
    cab = crudo[:fin].decode().splitlines()
    nv = int(next(l for l in cab if l.startswith("element vertex")).split()[-1])
    nf = int(next(l for l in cab if l.startswith("element face")).split()[-1])
    vt = np.dtype([("x", "<f4"), ("y", "<f4"), ("z", "<f4"), ("r", "u1"), ("g", "u1"), ("b", "u1")])
    v = np.frombuffer(crudo, vt, nv, fin)
    f = np.frombuffer(crudo, np.dtype([("n", "u1"), ("i", "<i4", 3)]), nf, fin + nv * vt.itemsize)
    return np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float64), np.stack([v["r"], v["g"], v["b"]], 1), f["i"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("malla")
    p.add_argument("--pasadas", type=int, default=10)
    a = p.parse_args()
    ruta = Path(a.malla)
    v, c, f = leer(ruta)
    n = len(v)
    i = np.concatenate([f[:, 0], f[:, 1], f[:, 2], f[:, 1], f[:, 2], f[:, 0]])
    j = np.concatenate([f[:, 1], f[:, 2], f[:, 0], f[:, 0], f[:, 1], f[:, 2]])
    A = coo_matrix((np.ones(len(i)), (i, j)), shape=(n, n)).tocsr()
    A.data[:] = 1
    L = diags(1 / np.maximum(A.sum(1).A1, 1)) @ A  # promedio de vecinos
    for _ in range(a.pasadas):
        v = v + 0.5 * (L @ v - v)
        v = v - 0.53 * (L @ v - v)
    salida = ruta.with_name(ruta.stem + "_suave.ply")
    guardar_malla_ply(salida, v.astype(np.float32), c, f)
    print(f"{n} vértices suavizados ({a.pasadas} pasadas) -> {salida}")


if __name__ == "__main__":
    main()
