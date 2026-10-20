"""Etapas B (máscaras 2D), C (elevación a 3D) y D (nombres). Ver docs/ia-nube.md §4.

B. Por cada vista, Grounding DINO da cajas para las frases del vocabulario y SAM 2 convierte cada caja
   en máscara. Las máscaras pequeñas se pintan encima de las grandes (un monitor sobre la mesa gana).
   Las fotos clave están separadas en el tiempo (no son un vídeo continuo), así que la identidad entre
   vistas no se propaga en 2D con el modo vídeo de SAM 2: se resuelve en 3D, que es más fiable.

C. Cada vértice candidato (no estructural) se proyecta en cada vista; se descarta si está tapado (profundidad
   LiDAR: |z − prof| > 3 cm, o el z-buffer de la propia malla donde no hay LiDAR) y vota por la máscara
   que lo cubre. La malla se divide en superpuntos (vóxel de 5 cm × dirección de la normal), que reciben la
   máscara mayoritaria de sus vértices en cada vista. Dos superpuntos vecinos se fusionan si en las vistas
   donde se ven ambos suelen caer en la misma máscara y rara vez en máscaras distintas (como SAI3D /
   Open3DIS). Las regiones que casi siempre están cubiertas por alguna máscara son instancias.

D. De cada instancia se recortan sus mejores vistas, se promedia el vector de SigLIP 2 y se compara con el
   vocabulario; se mezcla con los votos de Grounding DINO.
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np

from . import render
from .camaras import Vista
from .vocabulario import Entrada


INCIERTO = -3


@dataclass
class Mascara:
    vista: int
    frase: int
    puntuacion: float
    area: int


@dataclass
class Instancia:
    vertices: np.ndarray            # índices de vértice de la malla base
    entrada: int                    # índice en el vocabulario
    confianza: float
    alternativas: list = field(default_factory=list)   # [(índice, probabilidad)]
    vistas: int = 0


@dataclass
class Parametros:
    lado_superpunto: float = 0.05
    tolerancia_lidar: float = 0.03
    min_vertices: int = 300
    min_cubierto: float = 0.45       # fracción de las observaciones de la región cubiertas por una máscara
    min_acuerdo: int = 2
    min_razon_acuerdo: float = 0.6
    area_min: float = 0.0004         # fracción de la imagen
    area_max: float = 0.85
    iou_nms: float = 0.8
    vistas_nombre: int = 5


# ------------------------------------------------------------------------------------------- B. máscaras

def _nms(cajas, puntuaciones, umbral):
    orden = np.argsort(-puntuaciones)
    area = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1])
    quedan = []
    while len(orden):
        i = orden[0]
        quedan.append(i)
        x1 = np.maximum(cajas[i, 0], cajas[orden[1:], 0])
        y1 = np.maximum(cajas[i, 1], cajas[orden[1:], 1])
        x2 = np.minimum(cajas[i, 2], cajas[orden[1:], 2])
        y2 = np.minimum(cajas[i, 3], cajas[orden[1:], 3])
        inter = np.clip(x2 - x1, 0, None) * np.clip(y2 - y1, 0, None)
        iou = inter / (area[i] + area[orden[1:]] - inter + 1e-9)
        orden = orden[1:][iou < umbral]
    return np.asarray(quedan, int)


def mascaras_2d(vistas: list[Vista], entradas: list[Entrada], detector, segmentador,
                p: Parametros = Parametros(), informar=lambda f: None):
    """→ (mapa de ids por vista (H, W) int32: −1 sin máscara, INCIERTO en bordes; lista de Mascara con id = posición)."""
    from scipy import ndimage

    frases = [e.frase_deteccion for e in entradas]
    mapas, mascaras = [], []
    for vi, vista in enumerate(vistas):
        h, w = vista.alto, vista.ancho
        mapa = np.full((h, w), -1, np.int32)
        dets = detector.detectar(vista.imagen, frases)
        if dets:
            cajas = np.asarray([d[0] for d in dets], np.float32)
            punt = np.asarray([d[1] for d in dets], np.float32)
            frase = np.asarray([d[2] for d in dets], int)
            area = (cajas[:, 2] - cajas[:, 0]) * (cajas[:, 3] - cajas[:, 1]) / (w * h)
            ok = (area >= p.area_min) & (area <= p.area_max)
            cajas, punt, frase = cajas[ok], punt[ok], frase[ok]
            if len(cajas):
                quedan = _nms(cajas, punt, p.iou_nms)
                cajas, punt, frase = cajas[quedan], punt[quedan], frase[quedan]
                masks = segmentador.segmentar(vista.imagen, cajas)
                areas = masks.reshape(len(masks), -1).sum(1)
                for k in np.argsort(-areas):              # grandes primero, las pequeñas encima
                    if areas[k] < p.area_min * w * h:
                        continue
                    mapa[masks[k]] = len(mascaras)
                    mascaras.append(Mascara(vi, int(frase[k]), float(punt[k]), int(areas[k])))
        # Bordes de las máscaras (±2 px): un vértice ahí puede ser del objeto o de lo que hay detrás.
        borde = ndimage.maximum_filter(mapa, size=5) != ndimage.minimum_filter(mapa, size=5)
        mapa[borde] = INCIERTO
        mapas.append(mapa)
        informar((vi + 1) / len(vistas))
    return mapas, mascaras


# ------------------------------------------------------------------------------------------- C. elevación

def visibles(vista: Vista, posiciones: np.ndarray, caras: np.ndarray, separacion: float, tolerancia_lidar=0.03):
    """Vértices de la malla visibles en la vista → (índices, u, v) en píxeles enteros de la imagen de trabajo."""
    u, v, z = vista.proyectar(posiciones)
    w, h = vista.ancho, vista.alto
    ok = (z > 0.1) & (u >= 0) & (u <= w - 1) & (v >= 0) & (v <= h - 1)
    idx = np.nonzero(ok)[0]
    u, v, z = u[idx], v[idx], z[idx]
    ui, vi = np.round(u).astype(np.int64), np.round(v).astype(np.int64)

    referencia = np.full(len(idx), np.nan, np.float32)
    tolerancia = np.full(len(idx), np.nan, np.float32)
    pr = vista.profundidad
    if pr is not None:
        ph, pw = pr.mapa.shape
        pc = vista.a_camara(posiciones[idx])
        ud = np.round(pr.fx * pc[:, 0] / z + pr.cx).astype(np.int64)
        vd = np.round(pr.fy * pc[:, 1] / z + pr.cy).astype(np.int64)
        dentro = (ud >= 0) & (ud < pw) & (vd >= 0) & (vd < ph)
        d = np.zeros(len(idx), np.float32)
        d[dentro] = pr.mapa[vd[dentro], ud[dentro]]
        hay = dentro & (d > 0)
        referencia[hay] = d[hay]
        if vista.sintetica:   # profundidad exacta del z-buffer: solo el error de redondeo entre píxeles
            tolerancia[hay] = 0.01 + 0.005 * z[hay]
        else:                 # el LiDAR del iPhone tiene ~1 cm de ruido a 1 m y empeora con la distancia
            tolerancia[hay] = tolerancia_lidar + 0.01 * np.maximum(z[hay] - 1, 0)

    visible = np.ones(len(idx), bool)
    lidar = ~np.isnan(referencia)
    visible[lidar] = np.abs(z[lidar] - referencia[lidar]) <= tolerancia[lidar]
    falta = ~lidar
    if falta.any():
        # Donde no hay LiDAR: z-buffer de la propia malla a media resolución (más rápido y sin agujeros).
        s = 0.5
        zb, _ = render.zbuffer(posiciones, caras, vista.fx * s, vista.fy * s, (vista.cx + 0.5) * s - 0.5,
                               (vista.cy + 0.5) * s - 0.5, vista.mundo_a_cam,
                               int(np.ceil(w * s)), int(np.ceil(h * s)))
        us = np.clip(np.round((u[falta] + 0.5) * s - 0.5).astype(np.int64), 0, zb.shape[1] - 1)
        vs = np.clip(np.round((v[falta] + 0.5) * s - 0.5).astype(np.int64), 0, zb.shape[0] - 1)
        visible[falta] = z[falta] <= zb[vs, us] + 0.02 + 0.01 * z[falta] + separacion
    return idx[visible], ui[visible], vi[visible]


DIRECCIONES = np.array([(x, y, z) for x in (-1, 0, 1) for y in (-1, 0, 1) for z in (-1, 0, 1)
                        if (x, y, z) != (0, 0, 0)], np.float32)
DIRECCIONES /= np.linalg.norm(DIRECCIONES, axis=1, keepdims=True)


def superpuntos(posiciones: np.ndarray, normales: np.ndarray, lado: float) -> np.ndarray:
    """Regiones pequeñas y casi planas: mismo vóxel y misma dirección de normal (26 direcciones)."""
    if len(posiciones) == 0:
        return np.zeros(0, np.int64)
    direccion = np.argmax(normales @ DIRECCIONES.T, axis=1)
    voxel = np.floor(posiciones / lado).astype(np.int64)
    clave = np.column_stack([voxel, direccion])
    _, sp = np.unique(clave, axis=0, return_inverse=True)
    return sp.reshape(-1)


def _conexas(n, a, b):
    from scipy.sparse import coo_matrix
    from scipy.sparse.csgraph import connected_components

    g = coo_matrix((np.ones(len(a), np.int8), (a, b)), shape=(n, n))
    return connected_components(g, directed=False)[1]


def elevar(malla, candidatos: np.ndarray, vistas: list[Vista], mapas, mascaras: list[Mascara],
           n_entradas: int, p: Parametros = Parametros(), informar=lambda f: None):
    """→ (instancias candidatas [(vértices, votos DINO por entrada, nº de vistas cubiertas)],
          observaciones por vista [(vértices visibles, id de máscara)], estadísticas)."""
    pos = malla.posiciones
    separacion = render.separacion_media(pos, malla.caras)
    cand = np.nonzero(candidatos)[0]
    local = np.full(len(pos), -1, np.int64)
    local[cand] = np.arange(len(cand))
    sp = superpuntos(pos[cand], malla.normales[cand], p.lado_superpunto)
    n_sp = int(sp.max()) + 1 if len(sp) else 0
    tam = np.bincount(sp, minlength=n_sp)
    n_v = len(vistas)

    L = np.full((n_v, n_sp), -2, np.int32)          # −2 no visto, −1 visto sin máscara, ≥0 id de máscara
    observaciones = []
    for vi, vista in enumerate(vistas):
        idx, u, v = visibles(vista, pos, malla.caras, separacion, p.tolerancia_lidar)
        loc = local[idx]
        es_cand = loc >= 0
        idx, u, v, loc = idx[es_cand], u[es_cand], v[es_cand], loc[es_cand]
        m = mapas[vi][v, u].astype(np.int64)
        claro = m != INCIERTO
        idx, loc, m = idx[claro], loc[claro], m[claro]
        observaciones.append((idx, m))
        if len(loc) == 0:
            continue
        s = sp[loc]
        vistos = np.bincount(s, minlength=n_sp)
        clave = s * (len(mascaras) + 1) + (m + 1)
        claves, cuentas = np.unique(clave, return_counts=True)
        s_c, m_c = claves // (len(mascaras) + 1), claves % (len(mascaras) + 1) - 1
        orden = np.lexsort((-cuentas, s_c))           # por superpunto, la máscara más votada primero
        s_c, m_c, cuentas = s_c[orden], m_c[orden], cuentas[orden]
        primero = np.ones(len(s_c), bool)
        primero[1:] = s_c[1:] != s_c[:-1]
        s_c, m_c, cuentas = s_c[primero], m_c[primero], cuentas[primero]
        suficiente = vistos[s_c] >= np.minimum(3, tam[s_c])
        s_c, m_c, cuentas = s_c[suficiente], m_c[suficiente], cuentas[suficiente]
        L[vi, s_c] = np.where((m_c >= 0) & (cuentas >= 0.5 * vistos[s_c]), m_c, -1)
        informar((vi + 1) / n_v)

    # Vecindad entre superpuntos: aristas de la malla + centros cercanos (la malla del iPhone tiene huecos
    # y las esquinas de un mueble caen en superpuntos de normal distinta).
    from scipy.spatial import cKDTree

    caras = malla.caras
    aristas = np.concatenate([caras[:, [0, 1]], caras[:, [1, 2]], caras[:, [2, 0]]])
    la, lb = local[aristas[:, 0]], local[aristas[:, 1]]
    ok = (la >= 0) & (lb >= 0)
    a, b = sp[la[ok]], sp[lb[ok]]
    centros = np.zeros((n_sp, 3))
    if n_sp:
        np.add.at(centros, sp, pos[cand])
        centros /= tam[:, None]
        cercanos = cKDTree(centros).query_pairs(1.5 * p.lado_superpunto, output_type="ndarray")
        a, b = np.concatenate([a, cercanos[:, 0]]), np.concatenate([b, cercanos[:, 1]])
    distinto = a != b
    par = np.unique(np.sort(np.column_stack([a[distinto], b[distinto]]), axis=1), axis=0).reshape(-1, 2)
    acuerdo = np.zeros(len(par), np.int32)
    desacuerdo = np.zeros(len(par), np.int32)
    for vi in range(n_v):
        la_, lb_ = L[vi, par[:, 0]], L[vi, par[:, 1]]
        ambos = (la_ >= -1) & (lb_ >= -1)
        acuerdo += ambos & (la_ == lb_) & (la_ >= 0)
        desacuerdo += ambos & (la_ != lb_) & ((la_ >= 0) | (lb_ >= 0))
    min_acuerdo = p.min_acuerdo if n_v >= 6 else 1
    une = (acuerdo >= min_acuerdo) & (acuerdo >= p.min_razon_acuerdo * (acuerdo + desacuerdo))
    region = _conexas(n_sp, par[une, 0], par[une, 1]) if n_sp else np.zeros(0, np.int64)
    n_r = int(region.max()) + 1 if len(region) else 0

    # Cobertura de cada región y votos de Grounding DINO por entrada del vocabulario.
    visto = np.zeros(n_r)
    cubierto = np.zeros(n_r)
    vistas_cubiertas = np.zeros(n_r, np.int32)
    votos = np.zeros((n_r, n_entradas))
    frase = np.asarray([m.frase for m in mascaras], int)
    punt = np.asarray([m.puntuacion for m in mascaras], float)
    for vi in range(n_v):
        fila = L[vi]
        s_vistos = np.nonzero(fila >= -1)[0]
        np.add.at(visto, region[s_vistos], tam[s_vistos])
        s_cub = np.nonzero(fila >= 0)[0]
        np.add.at(cubierto, region[s_cub], tam[s_cub])
        vistas_cubiertas[np.unique(region[s_cub])] += 1
        np.add.at(votos, (region[s_cub], frase[fila[s_cub]]), tam[s_cub] * punt[fila[s_cub]])

    vert_region = region[sp] if len(sp) else np.zeros(0, np.int64)
    n_vert = np.bincount(vert_region, minlength=n_r)
    fraccion = cubierto / np.maximum(visto, 1)
    buenas = np.nonzero((fraccion >= p.min_cubierto) & (vistas_cubiertas >= min(2, n_v)) &
                        (n_vert >= p.min_vertices) & (votos.sum(1) > 0))[0]
    orden_vert = np.argsort(vert_region, kind="stable")
    cortes = np.searchsorted(vert_region[orden_vert], np.arange(n_r + 1))
    candidatas = []
    for r in buenas[np.argsort(-n_vert[buenas])]:
        vertices = cand[orden_vert[cortes[r]:cortes[r + 1]]]
        candidatas.append((vertices, votos[r], int(vistas_cubiertas[r])))
    stats = {"superpuntos": n_sp, "regiones": n_r, "fusiones": int(une.sum()), "separacion_media_m": separacion}
    return candidatas, observaciones, stats


# ------------------------------------------------------------------------------------------- D. nombres

def nombrar(candidatas, observaciones, vistas: list[Vista], posiciones, entradas: list[Entrada],
            nombrador, p: Parametros = Parametros(), informar=lambda f: None) -> list[Instancia]:
    from PIL import Image

    if not candidatas:
        return []
    n_i = len(candidatas)
    instancia = np.full(len(posiciones), -1, np.int64)
    for k, (vertices, _, _) in enumerate(candidatas):
        instancia[vertices] = k

    # Mejores vistas de cada instancia: donde más vértices suyos se ven cubiertos por alguna máscara.
    cuenta = np.zeros((len(vistas), n_i), np.int64)
    for vi, (idx, m) in enumerate(observaciones):
        k = instancia[idx]
        sel = (k >= 0) & (m >= 0)
        cuenta[vi] = np.bincount(k[sel], minlength=n_i)

    recortes, duenos = [], []
    for k in range(n_i):
        for vi in np.argsort(-cuenta[:, k])[:p.vistas_nombre]:
            if cuenta[vi, k] < 20:
                break
            vista = vistas[vi]
            idx, _ = observaciones[vi]
            propios = idx[instancia[idx] == k]
            u, v, _ = vista.proyectar(posiciones[propios])
            x0, x1, y0, y1 = np.percentile(u, 2), np.percentile(u, 98), np.percentile(v, 2), np.percentile(v, 98)
            margen_x = max(0.15 * (x1 - x0), (48 - (x1 - x0)) / 2, 0)
            margen_y = max(0.15 * (y1 - y0), (48 - (y1 - y0)) / 2, 0)
            caja = (int(max(0, x0 - margen_x)), int(max(0, y0 - margen_y)),
                    int(min(vista.ancho, x1 + margen_x + 1)), int(min(vista.alto, y1 + margen_y + 1)))
            if caja[2] - caja[0] < 8 or caja[3] - caja[1] < 8:
                continue
            recortes.append(Image.fromarray(vista.imagen).crop(caja))
            duenos.append(k)

    p_dino = np.stack([c[1] / max(c[1].sum(), 1e-9) for c in candidatas])
    prob = p_dino.copy()
    if recortes and nombrador is not None:
        texto = nombrador.vectores_texto([e.frase_nombre for e in entradas])
        imagen = np.concatenate([nombrador.vectores_imagen(recortes[i:i + 32]) for i in range(0, len(recortes), 32)])
        duenos = np.asarray(duenos)
        for k in range(n_i):
            sel = duenos == k
            if not sel.any():
                continue
            media = imagen[sel].mean(0)
            media /= np.linalg.norm(media) + 1e-9
            logits = nombrador.escala * (texto @ media)
            sig = np.exp(logits - logits.max())
            sig /= sig.sum()
            prob[k] = 0.75 * sig + 0.25 * p_dino[k]
            informar((k + 1) / n_i)

    instancias = []
    for k, (vertices, _, vistas_cub) in enumerate(candidatas):
        mejores = np.argsort(-prob[k])[:3]
        instancias.append(Instancia(vertices=vertices, entrada=int(mejores[0]), confianza=float(prob[k, mejores[0]]),
                                    alternativas=[(int(i), float(prob[k, i])) for i in mejores if prob[k, i] >= 0.01],
                                    vistas=vistas_cub))
    return instancias
