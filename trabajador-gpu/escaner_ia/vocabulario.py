"""Etapa A: vocabulario de búsqueda.

Grounding DINO solo entiende inglés, así que cada entrada lleva su traducción. SigLIP 2 es multilingüe:
las palabras del usuario que no están en la tabla se usan tal cual (en español) para nombrar y se pasan
también a Grounding DINO, que con suerte las reconoce.

`categoria` usa los mismos valores que la app (`SurfaceLabel.category`): mesa, asiento u objeto.
"""

from __future__ import annotations

import unicodedata
from dataclasses import dataclass

# Objetos de interior frecuentes (subconjunto de ScanNet200), sin paredes/suelo/techo/puertas/ventanas:
# esos los clasifica ARKit en el iPhone y se conservan de `escena.json`.
TABLA: list[tuple[str, str, str]] = [
    ("silla", "chair", "asiento"), ("sillón", "armchair", "asiento"), ("sofá", "sofa", "asiento"),
    ("taburete", "stool", "asiento"), ("banco", "bench", "asiento"), ("silla de oficina", "office chair", "asiento"),
    ("puf", "ottoman", "asiento"),
    ("mesa", "table", "mesa"), ("escritorio", "desk", "mesa"), ("mesa de centro", "coffee table", "mesa"),
    ("mesita de noche", "nightstand", "mesa"), ("mesa de comedor", "dining table", "mesa"),
    ("encimera", "kitchen counter", "mesa"),
    ("cama", "bed", "objeto"), ("almohada", "pillow", "objeto"), ("cojín", "cushion", "objeto"),
    ("manta", "blanket", "objeto"), ("colchón", "mattress", "objeto"),
    ("armario", "wardrobe", "objeto"), ("cómoda", "dresser", "objeto"), ("estantería", "bookshelf", "objeto"),
    ("estante", "shelf", "objeto"), ("mueble de cocina", "kitchen cabinet", "objeto"), ("cajonera", "drawer unit", "objeto"),
    ("mueble de TV", "tv stand", "objeto"), ("vitrina", "display cabinet", "objeto"),
    ("lámpara", "lamp", "objeto"), ("lámpara de pie", "floor lamp", "objeto"), ("lámpara de mesa", "table lamp", "objeto"),
    ("lámpara de techo", "ceiling lamp", "objeto"),
    ("televisor", "television", "objeto"), ("monitor", "computer monitor", "objeto"), ("ordenador portátil", "laptop", "objeto"),
    ("teclado", "keyboard", "objeto"), ("ratón", "computer mouse", "objeto"), ("ordenador", "computer tower", "objeto"),
    ("altavoz", "speaker", "objeto"), ("impresora", "printer", "objeto"), ("mando a distancia", "remote control", "objeto"),
    ("teléfono", "phone", "objeto"), ("videoconsola", "game console", "objeto"), ("router", "router", "objeto"),
    ("frigorífico", "refrigerator", "objeto"), ("horno", "oven", "objeto"), ("microondas", "microwave", "objeto"),
    ("lavavajillas", "dishwasher", "objeto"), ("lavadora", "washing machine", "objeto"), ("fregadero", "sink", "objeto"),
    ("cafetera", "coffee maker", "objeto"), ("tostadora", "toaster", "objeto"), ("hervidor", "kettle", "objeto"),
    ("vitrocerámica", "stove", "objeto"), ("campana extractora", "range hood", "objeto"),
    ("inodoro", "toilet", "objeto"), ("bañera", "bathtub", "objeto"), ("ducha", "shower", "objeto"),
    ("lavabo", "bathroom sink", "objeto"), ("espejo", "mirror", "objeto"), ("toalla", "towel", "objeto"),
    ("planta", "potted plant", "objeto"), ("maceta", "flower pot", "objeto"), ("jarrón", "vase", "objeto"),
    ("cuadro", "picture frame", "objeto"), ("reloj", "clock", "objeto"), ("cortina", "curtain", "objeto"),
    ("alfombra", "rug", "objeto"), ("persiana", "window blind", "objeto"), ("radiador", "radiator", "objeto"),
    ("ventilador", "fan", "objeto"), ("aire acondicionado", "air conditioner", "objeto"),
    ("papelera", "trash can", "objeto"), ("caja", "box", "objeto"), ("cesta", "basket", "objeto"),
    ("bolsa", "bag", "objeto"), ("mochila", "backpack", "objeto"), ("maleta", "suitcase", "objeto"),
    ("libro", "book", "objeto"), ("botella", "bottle", "objeto"), ("taza", "cup", "objeto"), ("vaso", "glass", "objeto"),
    ("plato", "plate", "objeto"), ("cuenco", "bowl", "objeto"), ("tabla de planchar", "ironing board", "objeto"),
    ("escalera de mano", "ladder", "objeto"), ("bicicleta", "bicycle", "objeto"), ("juguete", "toy", "objeto"),
    ("guitarra", "guitar", "objeto"), ("piano", "piano", "objeto"), ("perchero", "coat rack", "objeto"),
    ("zapatos", "shoes", "objeto"), ("ropa", "clothes", "objeto"), ("aspiradora", "vacuum cleaner", "objeto"),
    ("extintor", "fire extinguisher", "objeto"), ("enchufe", "power outlet", "objeto"), ("interruptor", "light switch", "objeto"),
    ("chimenea", "fireplace", "objeto"), ("columna", "column", "objeto"), ("escalera", "stairs", "objeto"),
]


@dataclass(frozen=True)
class Entrada:
    es: str
    en: str | None          # None → palabra del usuario sin traducción
    categoria: str

    @property
    def frase_deteccion(self) -> str:
        return (self.en or self.es).lower()

    @property
    def frase_nombre(self) -> str:
        return f"a photo of a {self.en}." if self.en else f"una foto de {self.es}."


def normalizar(texto: str) -> str:
    sin_tildes = unicodedata.normalize("NFKD", texto).encode("ascii", "ignore").decode()
    return " ".join(sin_tildes.lower().split())


def construir(extra_usuario: list[str] | None = None, etiquetas_iphone: list[str] | None = None) -> list[Entrada]:
    """Tabla fija + palabras del usuario + etiquetas que ya puso el iPhone (`etiquetaIA`), sin repetir."""
    entradas = [Entrada(es, en, cat) for es, en, cat in TABLA]
    por_nombre = {normalizar(e.es): e for e in entradas} | {normalizar(e.en): e for e in entradas}
    for palabra in (extra_usuario or []) + (etiquetas_iphone or []):
        clave = normalizar(palabra)
        if clave and clave not in por_nombre:
            nueva = Entrada(palabra.strip(), None, "objeto")
            entradas.append(nueva)
            por_nombre[clave] = nueva
    return entradas


def buscar(entradas: list[Entrada], texto: str) -> int | None:
    """Índice de la entrada que corresponde a una etiqueta devuelta por Grounding DINO."""
    t = normalizar(texto)
    if not t:
        return None
    frases = [normalizar(e.frase_deteccion) for e in entradas]
    if t in frases:
        return frases.index(t)
    # DINO a veces devuelve trozos ("office") o une frases ("chair table"): la frase más larga contenida.
    candidatos = [(len(f), i) for i, f in enumerate(frases) if f in t or t in f]
    return max(candidatos)[1] if candidatos else None
