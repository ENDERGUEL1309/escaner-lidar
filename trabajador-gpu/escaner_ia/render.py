"""Z-buffer de la malla: qué vértice se ve en cada píxel y a qué profundidad.

Sirve para dos cosas:
  • Prueba de oclusión cuando no hay profundidad LiDAR (o donde el LiDAR no da dato).
  • Modo de prueba sin fotos: se "fotografía" la malla coloreada desde cámaras virtuales.

Rasterizado sin OpenGL: cada triángulo se muestrea con una rejilla baricéntrica lo bastante fina para que
dos muestras vecinas estén a menos de 0,7 px (así no quedan huecos), con profundidad de perspectiva correcta
(1/z interpolado). Por píxel gana la muestra más cercana: se empaqueta (profundidad en décimas de mm << 32 |
índice del vértice más próximo de su triángulo) en un int64 y se toma el mínimo. La reducción se hace en la
GPU con PyTorch si está disponible; si no, con NumPy (más lento, pero sirve para las pruebas).
"""

from __future__ import annotations

from functools import lru_cache

import numpy as np

from .camaras import Vista

VACIO = np.iinfo(np.int64).max
PASO_PX = 0.7
NIVEL_MAX = 96
MUESTRAS_POR_LOTE = 4_000_000


def _torch():
    try:
        import torch
        return torch if torch.cuda.is_available() else None
    except Exception:
        return None


def separacion_media(posiciones: np.ndarray, caras: np.ndarray) -> float:
    if len(caras) == 0:
        return 0.01
    muestra = caras[np.random.default_rng(0).choice(len(caras), min(len(caras), 200_000), replace=False)]
    a, b = posiciones[muestra[:, 0]], posiciones[muestra[:, 1]]
    return float(np.median(np.linalg.norm(a - b, axis=1))) or 0.01


@lru_cache(maxsize=None)
def _rejilla(n: int):
    """Pesos baricéntricos (k, 3) de la rejilla de nivel n y, por muestra, el vértice de mayor peso."""
    i, j = np.meshgrid(np.arange(n + 1), np.arange(n + 1), indexing="ij")
    sel = (i + j) <= n
    w = np.column_stack([i[sel], j[sel], n - i[sel] - j[sel]]).astype(np.float32) / n
    return w, np.argmax(w, axis=1)


class _Reductor:
    def __init__(self, n_pix):
        self.torch = _torch()
        if self.torch is not None:
            self.clave = self.torch.full((n_pix,), VACIO, dtype=self.torch.int64, device="cuda")
        else:
            self.clave = np.full(n_pix, VACIO, np.int64)

    def añadir(self, pix: np.ndarray, val: np.ndarray):
        if len(pix) == 0:
            return
        if self.torch is not None:
            t = self.torch
            self.clave.scatter_reduce_(0, t.from_numpy(pix).to("cuda"), t.from_numpy(val).to("cuda"), reduce="amin")
            return
        orden = np.lexsort((val, pix))
        pix, val = pix[orden], val[orden]
        primero = np.ones(len(pix), bool)
        primero[1:] = pix[1:] != pix[:-1]
        pix, val = pix[primero], val[primero]
        self.clave[pix] = np.minimum(self.clave[pix], val)

    def resultado(self):
        return self.clave.cpu().numpy() if self.torch is not None else self.clave


def zbuffer(posiciones: np.ndarray, caras: np.ndarray, fx, fy, cx, cy, mundo_a_cam, ancho: int, alto: int,
            cerca: float = 0.05):
    """→ (profundidad (alto, ancho) float32 con inf donde no hay nada, índice de vértice (alto, ancho) con −1)."""
    r, t = mundo_a_cam[:3, :3].astype(np.float32), mundo_a_cam[:3, 3].astype(np.float32)
    pc = posiciones.astype(np.float32) @ r.T + t
    z = pc[:, 2]
    zs = np.where(z > cerca, z, 1.0)
    u = fx * pc[:, 0] / zs + cx
    v = fy * pc[:, 1] / zs + cy

    # Triángulos delante de la cámara y que tocan la imagen.
    tz = z[caras]
    tu, tv = u[caras], v[caras]
    ok = (tz > cerca).all(1)
    ok &= (tu.max(1) >= -0.5) & (tu.min(1) <= ancho - 0.5) & (tv.max(1) >= -0.5) & (tv.min(1) <= alto - 0.5)
    tri = np.nonzero(ok)[0]
    tu, tv, tiz, tid = tu[tri], tv[tri], 1.0 / tz[tri], caras[tri]

    def lados():
        return np.max(np.stack([np.hypot(tu[:, a] - tu[:, b], tv[:, a] - tv[:, b])
                                for a, b in ((0, 1), (1, 2), (2, 0))]), 0)

    # Triángulos enormes en pantalla (mallas gruesas o muy cerca de la cámara): se parten en 4 hasta que caben.
    lado = lados()
    for _ in range(8):
        grande = lado > NIVEL_MAX * PASO_PX
        if not grande.any():
            break
        g = [x[grande] for x in (tu, tv, tiz)]
        gid = tid[grande]
        medios = [[(x[:, a] + x[:, b]) / 2 for x in g] for a, b in ((0, 1), (1, 2), (2, 0))]
        esquinas = [[x[:, k] for x in g] for k in range(3)]
        m01, m12, m20 = medios
        hijos = ((esquinas[0], m01, m20, (0, 0, 2)), (m01, esquinas[1], m12, (0, 1, 1)),
                 (m20, m12, esquinas[2], (2, 1, 2)), (m01, m12, m20, (0, 1, 2)))
        nuevos = [[np.stack([h[0][c], h[1][c], h[2][c]], 1) for h in hijos] for c in range(3)]
        nid = [gid[:, list(h[3])] for h in hijos]
        tu = np.concatenate([tu[~grande]] + nuevos[0])
        tv = np.concatenate([tv[~grande]] + nuevos[1])
        tiz = np.concatenate([tiz[~grande]] + nuevos[2])
        tid = np.concatenate([tid[~grande]] + nid)
        lado = lados()
    nivel = np.clip(np.ceil(lado / PASO_PX), 1, NIVEL_MAX).astype(np.int64)

    red = _Reductor(ancho * alto)
    for n in np.unique(nivel):
        w, mayor = _rejilla(int(n))
        grupo = np.nonzero(nivel == n)[0]
        por_lote = max(1, MUESTRAS_POR_LOTE // len(w))
        for inicio in range(0, len(grupo), por_lote):
            g = grupo[inicio:inicio + por_lote]
            su = tu[g] @ w.T                                   # (T, k)
            sv = tv[g] @ w.T
            sz = 1.0 / (tiz[g] @ w.T)
            vert = tid[g][:, mayor]                            # vértice más próximo de cada muestra
            ui, vi = np.round(su).astype(np.int64), np.round(sv).astype(np.int64)
            dentro = (ui >= 0) & (ui < ancho) & (vi >= 0) & (vi < alto)
            pix = (vi * ancho + ui)[dentro]
            val = (np.round(sz[dentro] * 1e4).astype(np.int64) << 32) | vert[dentro].astype(np.int64)
            red.añadir(pix, val)

    clave = red.resultado()
    vacio = clave == VACIO
    prof = np.where(vacio, np.inf, (clave >> 32).astype(np.float64) / 1e4).astype(np.float32)
    indice = np.where(vacio, -1, clave & 0xFFFFFFFF).astype(np.int64)
    return prof.reshape(alto, ancho), indice.reshape(alto, ancho)


def fotografiar(posiciones, colores, caras, camaras) -> list[Vista]:
    """Modo de prueba sin fotos: imágenes sintéticas de la malla coloreada desde cámaras virtuales.
    La profundidad de cada vista es su propio z-buffer (sin ruido, como un LiDAR perfecto)."""
    from scipy import ndimage

    from .camaras import Profundidad

    vistas = []
    for nombre, fx, fy, cx, cy, m in camaras:
        ancho, alto = int(round((cx + 0.5) * 2)), int(round((cy + 0.5) * 2))
        prof, indice = zbuffer(posiciones, caras, fx, fy, cx, cy, m, ancho, alto)
        hay = indice >= 0
        if hay.mean() < 0.05:
            continue      # cámara mirando a la nada (fuera de la sala)
        # Agujeros de 1–2 px (bordes de triángulos o huecos de la malla): el píxel válido más cercano.
        distancia, (iy, ix) = ndimage.distance_transform_edt(~hay, return_indices=True)
        relleno = distancia <= 2
        imagen = np.full((alto, ancho, 3), 128, np.uint8)
        imagen[relleno] = colores[indice[iy, ix][relleno]]
        vistas.append(Vista(nombre=nombre, imagen=imagen, fx=fx, fy=fy, cx=cx, cy=cy, mundo_a_cam=m,
                            profundidad=Profundidad(np.where(np.isfinite(prof), prof, 0).astype(np.float32),
                                                    fx, fy, cx, cy),
                            sintetica=True))
    return vistas
