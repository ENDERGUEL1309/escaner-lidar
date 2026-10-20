"""Lectura y escritura de PLY binarios con el mismo formato que la app (ModelExporter.swift).

Vértices: x, y, z, nx, ny, nz, [s, t], red, green, blue [, inferido]   (float32 / uchar, little endian)
Caras:    property list uchar uint vertex_indices                        (solo triángulos)
"""

from __future__ import annotations

from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

_TIPOS = {
    "char": "i1", "int8": "i1", "uchar": "u1", "uint8": "u1",
    "short": "i2", "int16": "i2", "ushort": "u2", "uint16": "u2",
    "int": "i4", "int32": "i4", "uint": "u4", "uint32": "u4",
    "float": "f4", "float32": "f4", "double": "f8", "float64": "f8",
}


@dataclass
class Malla:
    posiciones: np.ndarray                    # (N, 3) float32
    normales: np.ndarray                      # (N, 3) float32
    colores: np.ndarray                       # (N, 3) uint8
    caras: np.ndarray                         # (M, 3) int64
    st: np.ndarray | None = None              # (N, 2) float32, ya en convención PLY (origen abajo)
    inferido: np.ndarray | None = None        # (N,) uint8
    textura: str | None = None                # "comment TextureFile …"
    extra: dict = field(default_factory=dict)

    @property
    def n_vertices(self) -> int:
        return len(self.posiciones)

    def submalla(self, triangulos: np.ndarray) -> "Malla":
        """Malla con solo esos triángulos y sus vértices (reindexados, en el orden en que aparecen)."""
        caras = self.caras[triangulos]
        usados, inversa = np.unique(caras.ravel(), return_inverse=True)
        return Malla(
            posiciones=self.posiciones[usados],
            normales=self.normales[usados],
            colores=self.colores[usados],
            caras=inversa.reshape(-1, 3).astype(np.int64),
            st=None if self.st is None else self.st[usados],
            inferido=None if self.inferido is None else self.inferido[usados],
            textura=self.textura,
        )


def leer(ruta: str | Path) -> Malla:
    datos = Path(ruta).read_bytes()
    fin = datos.find(b"end_header")
    if not datos.startswith(b"ply") or fin < 0:
        raise ValueError(f"{ruta}: no es un PLY")
    salto = datos.index(b"\n", fin) + 1
    cabecera = datos[:salto].decode("ascii", "replace").splitlines()

    formato, textura = None, None
    elementos: list[tuple[str, int, list]] = []
    for linea in cabecera:
        p = linea.split()
        if not p:
            continue
        if p[0] == "format":
            formato = p[1]
        elif p[0] == "comment" and len(p) >= 3 and p[1] == "TextureFile":
            textura = " ".join(p[2:])
        elif p[0] == "element":
            elementos.append((p[1], int(p[2]), []))
        elif p[0] == "property" and elementos:
            if p[1] == "list":
                elementos[-1][2].append((p[4], "list", _TIPOS[p[2]], _TIPOS[p[3]]))
            else:
                elementos[-1][2].append((p[2], _TIPOS[p[1]]))
    if formato != "binary_little_endian":
        raise ValueError(f"{ruta}: solo se admite PLY binary_little_endian (es {formato})")

    pos = salto
    vertices = None
    caras = np.zeros((0, 3), np.int64)
    for nombre, cuenta, props in elementos:
        if all(len(p) == 2 for p in props):
            tipo = np.dtype([(p[0], "<" + p[1]) for p in props])
            tabla = np.frombuffer(datos, tipo, cuenta, pos)
            pos += tipo.itemsize * cuenta
            if nombre == "vertex":
                vertices = tabla
        elif nombre == "face" and len(props) == 1:
            _, _, t_cuenta, t_indice = props[0]
            # Caso habitual (y el de la app): todas las caras son triángulos → lectura vectorizada.
            tipo = np.dtype([("n", "<" + t_cuenta), ("i", "<" + t_indice, 3)])
            if cuenta and np.frombuffer(datos, "<" + t_cuenta, 1, pos)[0] == 3:
                tabla = np.frombuffer(datos, tipo, cuenta, pos)
                if np.all(tabla["n"] == 3):
                    caras = tabla["i"].astype(np.int64)
                    pos += tipo.itemsize * cuenta
                    continue
            caras, pos = _caras_lentas(datos, pos, cuenta, t_cuenta, t_indice)
        else:
            raise ValueError(f"{ruta}: elemento PLY no admitido: {nombre}")

    if vertices is None:
        raise ValueError(f"{ruta}: no tiene vértices")
    nombres = vertices.dtype.names

    def columnas(*cols, tipo=np.float32):
        if not all(c in nombres for c in cols):
            return None
        return np.stack([vertices[c] for c in cols], axis=1).astype(tipo)

    n = len(vertices)
    return Malla(
        posiciones=columnas("x", "y", "z"),
        normales=columnas("nx", "ny", "nz") if "nx" in nombres else np.zeros((n, 3), np.float32),
        colores=columnas("red", "green", "blue", tipo=np.uint8) if "red" in nombres
        else np.full((n, 3), 180, np.uint8),
        caras=caras,
        st=columnas("s", "t"),
        inferido=vertices["inferido"].astype(np.uint8) if "inferido" in nombres else None,
        textura=textura,
    )


def _caras_lentas(datos, pos, cuenta, t_cuenta, t_indice):
    """Caras de tamaño variable (polígonos): se triangulan en abanico."""
    tc, ti = np.dtype("<" + t_cuenta), np.dtype("<" + t_indice)
    tris = []
    for _ in range(cuenta):
        k = int(np.frombuffer(datos, tc, 1, pos)[0])
        pos += tc.itemsize
        idx = np.frombuffer(datos, ti, k, pos)
        pos += ti.itemsize * k
        for j in range(1, k - 1):
            tris.append((idx[0], idx[j], idx[j + 1]))
    return np.asarray(tris, np.int64).reshape(-1, 3), pos


def escribir(malla: Malla, ruta: str | Path, inferido: bool = True) -> None:
    """Mismo orden de propiedades que la app; con `inferido` añade `property uchar inferido` al final."""
    n = malla.n_vertices
    con_st = malla.st is not None
    cab = ["ply", "format binary_little_endian 1.0", "comment Escaner LiDAR - IA en la nube"]
    if con_st and malla.textura:
        cab.append(f"comment TextureFile {malla.textura}")
    cab.append(f"element vertex {n}")
    cab += [f"property float {c}" for c in ("x", "y", "z", "nx", "ny", "nz")]
    if con_st:
        cab += ["property float s", "property float t"]
    cab += ["property uchar red", "property uchar green", "property uchar blue"]
    if inferido:
        cab.append("property uchar inferido")
    cab += [f"element face {len(malla.caras)}", "property list uchar uint vertex_indices", "end_header"]

    campos = [(c, "<f4") for c in ("x", "y", "z", "nx", "ny", "nz")]
    if con_st:
        campos += [("s", "<f4"), ("t", "<f4")]
    campos += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
    if inferido:
        campos.append(("inferido", "u1"))
    v = np.empty(n, np.dtype(campos))
    for i, c in enumerate(("x", "y", "z")):
        v[c] = malla.posiciones[:, i]
    for i, c in enumerate(("nx", "ny", "nz")):
        v[c] = malla.normales[:, i]
    if con_st:
        v["s"], v["t"] = malla.st[:, 0], malla.st[:, 1]
    for i, c in enumerate(("red", "green", "blue")):
        v[c] = malla.colores[:, i]
    if inferido:
        v["inferido"] = 0 if malla.inferido is None else malla.inferido

    f = np.empty(len(malla.caras), np.dtype([("n", "u1"), ("i", "<u4", 3)]))
    f["n"] = 3
    f["i"] = malla.caras

    ruta = Path(ruta)
    ruta.parent.mkdir(parents=True, exist_ok=True)
    with open(ruta, "wb") as archivo:
        archivo.write(("\n".join(cab) + "\n").encode("ascii"))
        archivo.write(v.tobytes())
        archivo.write(f.tobytes())


def unir(mallas: list[Malla]) -> tuple[Malla, np.ndarray]:
    """Concatena mallas; devuelve la unión y, por triángulo, el índice de la malla de origen."""
    desplazamiento = 0
    caras, origen = [], []
    for i, m in enumerate(mallas):
        caras.append(m.caras + desplazamiento)
        origen.append(np.full(len(m.caras), i, np.int32))
        desplazamiento += m.n_vertices
    con_st = all(m.st is not None for m in mallas)
    union = Malla(
        posiciones=np.concatenate([m.posiciones for m in mallas]),
        normales=np.concatenate([m.normales for m in mallas]),
        colores=np.concatenate([m.colores for m in mallas]),
        caras=np.concatenate(caras) if caras else np.zeros((0, 3), np.int64),
        st=np.concatenate([m.st for m in mallas]) if con_st else None,
        textura=next((m.textura for m in mallas if m.textura), None),
    )
    return union, np.concatenate(origen) if origen else np.zeros(0, np.int32)
