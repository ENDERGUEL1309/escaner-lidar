r"""Segmentación semántica de cada foto con SAM3 (Segment Anything 3, de Meta): qué es cada píxel.

    %USERPROFILE%\Herramientas\SAM3\.venv\Scripts\python segmentar_fotos.py "C:\…\Habitacion_…"

SAM3 encuentra todas las instancias de un concepto escrito («sofa», «wall», «window»…). Para cada foto se pide la
lista de conceptos de abajo, más los muebles que RoomPlan vio en habitacion.json, y cada píxel se queda con el
concepto de la máscara más segura que lo cubre. Los objetos ganan a las superficies (un cojín delante de la pared es
cojín). Después votar_etiquetas.py lleva estas etiquetas a cada punto del LiDAR comparando todas las vistas, como
en la segmentación por splat de 3DGS: los fallos de una foto suelta se corrigen con las demás.

Resultado en <escaneo>/segmentacion/: clases.json y, por foto, <carpeta>/<foto>.png en modo LA
(L = índice de la clase, 0 = sin etiqueta; A = confianza de 0 a 255).

Necesita acceso a facebook/sam3 en Hugging Face (modelo restringido: aceptar la licencia y `hf auth login`).
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import numpy as np
import torch
from PIL import Image

# (nombre en español, concepto para SAM3, es superficie). El índice en la lista + 1 es la etiqueta en las PNG.
CLASES = [
    ("pared", "wall", True), ("suelo", "floor", True), ("techo", "ceiling", True),
    ("ventana", "window", True), ("puerta", "door", True), ("escalera", "staircase", True),
    ("sofá", "sofa", False), ("silla", "chair", False), ("mesa", "table", False),
    ("escritorio", "desk", False), ("cama", "bed", False), ("armario", "cabinet", False),
    ("estantería", "shelf", False), ("televisor", "television", False),
    ("electrodoméstico", "appliance", False), ("nevera", "refrigerator", False),
    ("planta", "plant", False), ("lámpara", "lamp", False), ("alfombra", "rug", False),
    ("cojín", "cushion", False), ("cuadro", "picture frame", False), ("persona", "person", False),
]
# Tipos de mueble de RoomPlan (habitacion.json, campo kind) → concepto de SAM3 que hay que pedir sí o sí.
ROOMPLAN = {"sofa": "sofa", "chair": "chair", "table": "table", "desk": "desk", "bed": "bed",
            "storage": "cabinet", "appliance": "appliance", "television": "television",
            "refrigerator": "refrigerator", "stairs": "staircase"}


def fotos(origen: Path):
    for sub in ("fotos_clave", "fotogramas"):
        indice = origen / sub / "camaras.json"
        if indice.exists():
            for f in json.loads(indice.read_text("utf-8"))["fotos"]:
                yield sub, f["archivo"]


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    p.add_argument("--umbral", type=float, default=0.4, help="confianza mínima de una máscara de SAM3")
    p.add_argument("--modelo", default="facebook/sam3")
    a = p.parse_args()
    origen = Path(a.carpeta).resolve()
    destino = origen / "segmentacion"
    destino.mkdir(exist_ok=True)

    clases = list(CLASES)
    sala = origen / "habitacion.json"
    if sala.exists():
        conceptos = {c[1] for c in clases}
        for m in json.loads(sala.read_text("utf-8")).get("furniture", []):
            concepto = ROOMPLAN.get(m.get("kind"), m.get("kind"))
            if concepto and concepto not in conceptos:
                clases.append((m.get("name", concepto), concepto, False))
                conceptos.add(concepto)
    (destino / "clases.json").write_text(json.dumps(
        [{"etiqueta": i + 1, "nombre": n, "concepto": c, "superficie": s} for i, (n, c, s) in enumerate(clases)],
        indent=1, ensure_ascii=False), "utf-8")

    from transformers import Sam3Model, Sam3Processor
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    modelo = Sam3Model.from_pretrained(a.modelo, dtype=torch.float16 if dev.type == "cuda" else torch.float32)
    modelo = modelo.to(dev).eval()
    proc = Sam3Processor.from_pretrained(a.modelo)
    # Los textos se codifican una vez: cambian los conceptos, no las fotos.
    textos = [proc(text=c, return_tensors="pt").to(dev) for _, c, _ in clases]

    lista = list(fotos(origen))
    print(f"SAM3: {len(lista)} fotos, {len(clases)} conceptos", flush=True)
    t0 = time.time()
    hechas = 0
    for sub, archivo in lista:
        salida = destino / sub / (Path(archivo).stem + ".png")
        if salida.exists():
            continue
        salida.parent.mkdir(parents=True, exist_ok=True)
        img = Image.open(origen / sub / archivo).convert("RGB")
        entrada = proc(images=img, return_tensors="pt").to(dev)
        with torch.no_grad():
            vision = modelo.get_vision_features(pixel_values=entrada.pixel_values.to(modelo.dtype))
            mejor = torch.zeros(img.height, img.width, device=dev)
            etiqueta = torch.zeros(img.height, img.width, dtype=torch.uint8, device=dev)
            for k, ((_, _, superficie), texto) in enumerate(zip(clases, textos)):
                salida_modelo = modelo(vision_embeds=vision, **texto)
                r = proc.post_process_instance_segmentation(
                    salida_modelo, threshold=a.umbral, mask_threshold=0.5,
                    target_sizes=entrada.get("original_sizes").tolist())[0]
                for mascara, puntuacion in zip(r["masks"], r["scores"]):
                    # Una superficie solo gana donde no hay un objeto igual de seguro delante.
                    s = float(puntuacion) * (0.8 if superficie else 1.0)
                    gana = mascara.to(dev).bool() & (s > mejor)
                    mejor[gana] = s
                    etiqueta[gana] = k + 1
        la = np.stack([etiqueta.cpu().numpy(), (mejor.clamp(0, 1) * 255).byte().cpu().numpy()], -1)
        Image.fromarray(la, "LA").save(salida, optimize=True)
        hechas += 1
        if hechas % 10 == 0:
            print(f"  {hechas} fotos ({(time.time() - t0) / hechas:.1f} s por foto)", flush=True)
    print(f"Segmentación lista en {(time.time() - t0) / 60:.1f} min → {destino}", flush=True)


if __name__ == "__main__":
    main()
