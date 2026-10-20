"""Proceso completo de un escaneo: carpeta de entrada → carpeta de salida (escena.json v2 + objetos/*.ply).

Independiente de Modal: se puede ejecutar con modelos falsos en un PC sin GPU (ver pruebas/).
"""

from __future__ import annotations

import json
import math
import shutil
import time
import unicodedata
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from . import camaras, ply, render, segmentacion, vocabulario

SERVICIO = "escaner-lidar-ia"
VERSION = "0.1.0"

ETAPAS = {
    "descarga": "Descargando el escaneo",
    "vocabulario": "Buscando qué hay en las fotos",
    "mascaras": "Segmentando objetos en las fotos",
    "elevacion3d": "Proyectando máscaras sobre la malla",
    "nombres": "Nombrando objetos",
    "completado": "Completando la geometría oculta",
    "splats": "Entrenando Gaussian Splatting",
    "empaquetado": "Preparando resultados",
}


def etapas_para(opciones: dict) -> list[str]:
    """Mismas etapas y en el mismo orden que `etapasPara` de servidor/src/trabajador.js."""
    etapas = ["descarga"]
    if opciones.get("segmentacion", True) is not False:
        etapas += ["vocabulario", "mascaras", "elevacion3d", "nombres"]
    if opciones.get("completado"):
        etapas.append("completado")
    if opciones.get("splats"):
        etapas.append("splats")
    return etapas + ["empaquetado"]


class Progreso:
    """Traduce (etapa, fracción dentro de la etapa) a progreso global 0–1 y lo envía con `enviar`."""

    def __init__(self, etapas: list[str], enviar, intervalo=2.0):
        self.etapas, self.enviar, self.intervalo = etapas, enviar, intervalo
        self.actual, self.ultimo, self.tiempos, self._t0 = None, 0.0, {}, None

    def etapa(self, clave: str):
        ahora = time.time()
        if self.actual is not None:
            self.tiempos[self.actual] = round(ahora - self._t0, 2)
        self.actual, self._t0 = clave, ahora
        self.enviar(ETAPAS[clave], self.etapas.index(clave) / len(self.etapas))
        self.ultimo = ahora

    def avance(self, fraccion: float):
        ahora = time.time()
        if ahora - self.ultimo >= self.intervalo:
            i = self.etapas.index(self.actual)
            self.enviar(ETAPAS[self.actual], (i + min(max(fraccion, 0), 1)) / len(self.etapas))
            self.ultimo = ahora

    def cerrar(self):
        if self.actual is not None:
            self.tiempos[self.actual] = round(time.time() - self._t0, 2)


# ----------------------------------------------------------------------------------------- utilidades

def nombre_archivo(nombre: str) -> str:
    """Igual que `ModelExporter.objectName` de la app."""
    plano = unicodedata.normalize("NFKD", nombre).encode("ascii", "ignore").decode()
    limpio = "".join(c if c.isascii() and c.isalnum() else "_" for c in plano)
    return "_" + limpio if limpio[:1].isdigit() else limpio


def caja_orientada(posiciones: np.ndarray):
    """Igual que `SceneSegmenter.info`: caja con giro solo alrededor de Y. → centro, dimensiones, rotacionY (°)."""
    p = posiciones.astype(np.float64)
    media = p.mean(0)
    dx, dz = p[:, 0] - media[0], p[:, 2] - media[2]
    angulo = 0.5 * math.atan2(2 * (dx * dz).sum(), (dx * dx).sum() - (dz * dz).sum())
    c, s = math.cos(angulo), math.sin(angulo)
    local = np.column_stack([dx * c + dz * s, p[:, 1], -dx * s + dz * c])
    lo, hi = local.min(0), local.max(0)
    tam, mitad = hi - lo, (lo + hi) / 2
    centro = [media[0] + mitad[0] * c - mitad[2] * s, mitad[1], media[2] + mitad[0] * s + mitad[2] * c]
    return [round(float(x), 4) for x in centro], [round(float(x), 4) for x in tam], round(math.degrees(angulo), 3)


def triangulos_por_etiqueta(caras: np.ndarray, etiqueta_vertice: np.ndarray) -> np.ndarray:
    """Cada triángulo va con la etiqueta de la mayoría de sus vértices (misma regla que la app)."""
    e0, e1, e2 = (etiqueta_vertice[caras[:, i]] for i in range(3))
    return np.where(e1 == e2, e1, e0)


def info_objeto(nombre, categoria, estructura, malla: ply.Malla, archivo, **extra):
    centro, dim, rot = caja_orientada(malla.posiciones)
    return {"nombre": nombre, "categoria": categoria, "estructura": estructura, "centro": centro,
            "dimensiones": dim, "rotacionY": rot, "vertices": int(malla.n_vertices),
            "triangulos": int(len(malla.caras)), "archivo": archivo, **extra}


# ----------------------------------------------------------------------------------------- entrada

class Escaneo:
    def __init__(self, carpeta: Path, avisos: list[str]):
        self.carpeta = carpeta
        self.escena = None
        if (carpeta / "escena.json").exists():
            self.escena = json.loads((carpeta / "escena.json").read_text("utf-8"))

        # Partes del iPhone (con textura si la hay). Juntas forman la malla texturizada completa.
        self.partes: list[tuple[dict, ply.Malla]] = []
        for obj in (self.escena or {}).get("objetos", []):
            ruta = carpeta / obj.get("archivo", "")
            if obj.get("archivo") and ruta.is_file():
                self.partes.append((obj, ply.leer(ruta)))
            else:
                avisos.append(f"escena.json cita {obj.get('archivo')!r}, que no se ha subido")
        self.union = None
        if self.partes:
            self.union, self.origen = ply.unir([m for _, m in self.partes])
            self.parte_vertice = np.zeros(self.union.n_vertices, np.int32)
            self.parte_vertice[self.union.caras.ravel()] = np.repeat(self.origen, 3)

        if (carpeta / "malla_color.ply").exists():
            self.base = ply.leer(carpeta / "malla_color.ply")
        elif self.union is not None:
            self.base = self.union
        else:
            raise ValueError("El escaneo no tiene malla_color.ply ni objetos/*.ply")

    def estructurales(self) -> np.ndarray:
        """Vértices de la malla base que son paredes, suelo, techo, puertas o ventanas según el iPhone."""
        if self.union is None:
            return np.zeros(self.base.n_vertices, bool)
        estructura = np.array([bool(o.get("estructura")) for o, _ in self.partes])
        if self.base is self.union:
            return estructura[self.parte_vertice]
        from scipy.spatial import cKDTree

        d, j = cKDTree(self.union.posiciones).query(self.base.posiciones, distance_upper_bound=0.03)
        cerca = np.isfinite(d)
        resultado = np.zeros(self.base.n_vertices, bool)
        resultado[cerca] = estructura[self.parte_vertice[j[cerca]]]
        return resultado


def cargar_vistas(carpeta: Path, base: ply.Malla, modo: str, avisos: list[str]):
    """Fotos reales si hay `fotos_clave/camaras.json` (y el modo no es "sinteticas"); si no, cámaras virtuales."""
    fotos = carpeta / "fotos_clave"
    if modo != "sinteticas" and (fotos / "camaras.json").exists():
        vistas, av = camaras.cargar_fotos(fotos)
        avisos += av
        if vistas:
            return vistas, "fotos"
        avisos.append("camaras.json no tiene fotos utilizables")
    if modo == "fotos":
        raise ValueError("Se pidieron fotos reales y el escaneo no trae fotos_clave/camaras.json")
    avisos.append("Modo de prueba sin fotos: se usan imágenes sintéticas de la malla coloreada desde "
                  "cámaras virtuales. La detección será peor que con las fotos de 12 MP del iPhone.")
    virtuales = camaras.camaras_virtuales(base.posiciones)
    return render.fotografiar(base.posiciones, base.colores, base.caras, virtuales), "sinteticas"


# ----------------------------------------------------------------------------------------- proceso

def procesar(entrada: Path, salida: Path, opciones: dict, modelos, enviar=lambda etapa, progreso: None,
             parametros: segmentacion.Parametros | None = None) -> dict:
    """Ejecuta las etapas A–D y escribe la salida. `enviar(etapa, progreso)` informa del avance.
    La etapa «descarga» la hace quien llama; aquí se marca como hecha. Devuelve el informe."""
    entrada, salida = Path(entrada), Path(salida)
    p = parametros or segmentacion.Parametros()
    etapas = etapas_para(opciones)
    progreso = Progreso(etapas, enviar)
    avisos: list[str] = []
    inicio = time.time()
    progreso.etapa("descarga")

    escaneo = Escaneo(entrada, avisos)
    base = escaneo.base
    informe = {"servicio": SERVICIO, "version": VERSION, "opciones": opciones, "avisos": avisos,
               "modelos": [], "malla": {"vertices": base.n_vertices, "triangulos": int(len(base.caras))}}

    instancias, entradas = [], []
    if "vocabulario" in etapas:
        progreso.etapa("vocabulario")
        etiquetas_iphone = [o.get("etiquetaIA") for o, _ in escaneo.partes if o.get("etiquetaIA")]
        entradas = vocabulario.construir(opciones.get("vocabulario"), etiquetas_iphone)
        vistas, tipo = cargar_vistas(entrada, base, opciones.get("vistas", "auto"), avisos)
        informe["vistas"] = {"tipo": tipo, "numero": len(vistas)}
        informe["vocabulario"] = len(entradas)

        progreso.etapa("mascaras")
        mapas, mascaras = segmentacion.mascaras_2d(vistas, entradas, modelos.detector, modelos.segmentador, p,
                                                   progreso.avance)
        informe["mascaras2d"] = len(mascaras)

        progreso.etapa("elevacion3d")
        candidatos = ~escaneo.estructurales()
        candidatas, observaciones, stats = segmentacion.elevar(base, candidatos, vistas, mapas, mascaras,
                                                               len(entradas), p, progreso.avance)
        informe["elevacion"] = stats

        progreso.etapa("nombres")
        instancias = segmentacion.nombrar(candidatas, observaciones, vistas, base.posiciones, entradas,
                                          modelos.nombrador if candidatas else None, p, progreso.avance)
        informe["modelos"] = list(getattr(modelos, "versiones", []))
        del mapas, observaciones, vistas

    for clave in ("completado", "splats"):
        if clave in etapas:
            progreso.etapa(clave)
            avisos.append(f"La etapa «{ETAPAS[clave]}» todavía no está implementada; se omite.")

    progreso.etapa("empaquetado")
    objetos = empaquetar(escaneo, instancias, entradas, salida, segmentar="vocabulario" in etapas)
    progreso.cerrar()
    informe.update(instancias=len(instancias), objetos=len(objetos), tiempos=progreso.tiempos,
                   duracion=round(time.time() - inicio, 2))
    (salida / "informe.json").write_text(json.dumps(informe, ensure_ascii=False, indent=2), "utf-8")
    return informe


def empaquetar(escaneo: Escaneo, instancias, entradas, salida: Path, segmentar=True) -> list[dict]:
    salida.mkdir(parents=True, exist_ok=True)
    if (salida / "objetos").exists():
        shutil.rmtree(salida / "objetos")
    (salida / "objetos").mkdir()
    textura = escaneo.carpeta / "textura.jpg"
    if textura.exists():
        shutil.copyfile(textura, salida / "textura.jpg")

    objetos: list[dict] = []
    usados: set[str] = set()

    def guardar(nombre, malla: ply.Malla, **campos):
        archivo = nombre_archivo(nombre)
        base_archivo, n = archivo, 2
        while archivo.lower() in usados:      # Windows/macOS no distinguen mayúsculas
            archivo, n = f"{base_archivo}_{n}", n + 1
        usados.add(archivo.lower())
        if malla.st is not None and textura.exists():
            malla.textura = "../textura.jpg"
        ply.escribir(malla, salida / "objetos" / f"{archivo}.ply")
        objetos.append(info_objeto(nombre, malla=malla, archivo=f"objetos/{archivo}.ply", **campos))

    union = escaneo.union
    # 1. Estructura del iPhone tal cual (ARKit la clasifica bien y la app ya la aplanó).
    if union is not None:
        for i, (obj, _) in enumerate(escaneo.partes):
            if obj.get("estructura") or not segmentar:
                extra = {k: v for k, v in obj.items() if k in ("etiquetaIA", "confianzaIA")}
                guardar(obj["nombre"], union.submalla(np.nonzero(escaneo.origen == i)[0]),
                        categoria=obj.get("categoria", "objeto"), estructura=bool(obj.get("estructura")),
                        fuente="iphone", fraccionInferida=0, **extra)
    elif not segmentar:
        guardar("Escaneo completo", escaneo.base, categoria="objeto", estructura=False, fuente="iphone",
                fraccionInferida=0)

    if segmentar:
        # 2. Objetos de la nube: etiqueta por vértice de la malla base → triángulos de la malla con textura.
        base = escaneo.base
        etiqueta = np.full(base.n_vertices, -1, np.int64)
        for k, inst in enumerate(instancias):
            etiqueta[inst.vertices] = k
        if union is not None:
            estructura_parte = np.array([bool(o.get("estructura")) for o, _ in escaneo.partes])
            tri_no_estructura = np.nonzero(~estructura_parte[escaneo.origen])[0]
            if base is union:
                etiqueta_union = etiqueta
            else:
                from scipy.spatial import cKDTree

                _, j = cKDTree(base.posiciones).query(union.posiciones)
                etiqueta_union = etiqueta[j]
            destino, tri_ids = union, tri_no_estructura
            tri_etiqueta = triangulos_por_etiqueta(union.caras[tri_ids], etiqueta_union)
        else:
            destino, tri_ids = base, np.arange(len(base.caras))
            tri_etiqueta = triangulos_por_etiqueta(base.caras, etiqueta)

        contadores: dict[str, int] = {}
        orden = sorted(range(len(instancias)), key=lambda k: -len(instancias[k].vertices))
        for k in orden:
            inst = instancias[k]
            tris = tri_ids[tri_etiqueta == k]
            if len(tris) == 0:
                continue
            e = entradas[inst.entrada]
            base_nombre = e.es[:1].upper() + e.es[1:]
            contadores[base_nombre] = contadores.get(base_nombre, 0) + 1
            guardar(f"{base_nombre} {contadores[base_nombre]}", destino.submalla(tris),
                    categoria=e.categoria, estructura=False,
                    etiquetaIA=e.es, confianzaIA=round(inst.confianza, 3), fuente="nube",
                    alternativas=[[entradas[i].es, round(pr, 3)] for i, pr in inst.alternativas],
                    fraccionInferida=0, vistas=inst.vistas)
        resto = tri_ids[tri_etiqueta < 0]
        if len(resto):
            guardar("Fragmentos", destino.submalla(resto), categoria="fragmentos", estructura=False,
                    fuente="nube", fraccionInferida=0)

    original = escaneo.escena or {}
    escena = {k: v for k, v in original.items() if k != "objetos"}
    escena.update(version=2, unidades=original.get("unidades", "metros"), ejeVertical=original.get("ejeVertical", "Y"))
    escena["procesado"] = {"servicio": SERVICIO, "version": VERSION,
                           "fecha": datetime.now(timezone.utc).replace(microsecond=0).isoformat().replace("+00:00", "Z"),
                           "modelos": ["grounding-dino-base", "sam2.1-hiera-large", "siglip2-so400m"] if segmentar else []}
    escena["objetos"] = objetos
    (salida / "escena.json").write_text(json.dumps(escena, ensure_ascii=False, indent=2), "utf-8")
    return objetos
