r"""Remodelado de la habitación, paso 3: forma 3D de cada mueble a partir de su foto recortada (Hunyuan3D-2).

Se ejecuta con el Python de Hunyuan3D-2 (su entorno es aparte para no romper .venv-splat):

    %USERPROFILE%\Herramientas\Hunyuan3D-2\.venv\Scripts\python generar_muebles.py "C:\…\Habitacion_…"

Lee remodelado/objetos/objetos.json y cada mueble_N.png (fondo transparente). Los muebles escaneados en detalle con
Object Capture y los que no tienen una vista completa se saltan. Resultado: remodelado/objetos/mueble_N_ia.glb, una
forma sin escala ni orientación fijas: ajustar_muebles.py la encaja en la caja medida del mueble y la comprueba
contra el LiDAR.
"""

from __future__ import annotations

import argparse
import json
import time
from pathlib import Path

import torch
from PIL import Image


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    p.add_argument("--pasos", type=int, default=30)
    p.add_argument("--resolucion", type=int, default=320, help="resolución del octree (más = más detalle y tiempo)")
    a = p.parse_args()
    carpeta = Path(a.carpeta).resolve() / "remodelado" / "objetos"
    info = json.loads((carpeta / "objetos.json").read_text("utf-8"))
    pendientes = [m for m in info["muebles"] if m["recorte"] and not m["detalle"]
                  and not (carpeta / f"{m['clave']}_ia.glb").exists()]
    if not pendientes:
        print("Generación 3D: nada pendiente", flush=True)
        return

    from hy3dgen.shapegen import Hunyuan3DDiTFlowMatchingPipeline

    t0 = time.time()
    pipeline = Hunyuan3DDiTFlowMatchingPipeline.from_pretrained("tencent/Hunyuan3D-2", subfolder="hunyuan3d-dit-v2-0",
                                                                variant="fp16")
    print(f"Generación 3D: modelo cargado en {time.time() - t0:.0f} s; {len(pendientes)} muebles", flush=True)
    for i, m in enumerate(pendientes, 1):
        t0 = time.time()
        imagen = Image.open(carpeta / f"{m['clave']}.png").convert("RGBA")
        malla = pipeline(image=imagen, num_inference_steps=a.pasos, octree_resolution=a.resolucion, num_chunks=20000,
                         generator=torch.manual_seed(0), output_type="trimesh")[0]
        malla.export(carpeta / f"{m['clave']}_ia.glb")
        print(f"Generación 3D {i}/{len(pendientes)}: {m['nombre']} en {time.time() - t0:.0f} s "
              f"({len(malla.faces)} caras)", flush=True)
        torch.cuda.empty_cache()


if __name__ == "__main__":
    main()
