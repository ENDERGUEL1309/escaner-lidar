"""Fotogrametría de alto detalle en este PC (GPU NVIDIA local), sin subir nada a la nube.

    .venv\\Scripts\\python fotogrametria_local.py "C:\\…\\Entorno_…"

Necesita COLMAP (versión CUDA) y OpenMVS para Windows; por defecto los busca en C:\\Users\\<usuario>\\Herramientas.
El trabajo intermedio va a Herramientas\\trabajos\\<escaneo> (se puede reanudar: la extracción de puntos no se repite)
y el resultado a <escaneo>\\fotogrametria\\.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from escaner_ia import fotogrametria  # noqa: E402

HERRAMIENTAS = Path.home() / "Herramientas"


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta", help="carpeta del escaneo (con fotos_clave/ y fotogramas/)")
    p.add_argument("--max-fotogramas", type=int, default=160)
    p.add_argument("--solo-openmvs", action="store_true", help="reanudar desde la malla (COLMAP ya terminado)")
    p.add_argument("--colmap", default=str(HERRAMIENTAS / "colmap" / "COLMAP.bat"))
    p.add_argument("--openmvs", default=str(HERRAMIENTAS / "openmvs" / "vc17" / "x64" / "Release"))
    a = p.parse_args()

    origen = Path(a.carpeta).resolve()
    trabajo = HERRAMIENTAS / "trabajos" / origen.name
    destino = origen / "fotogrametria"
    destino.mkdir(exist_ok=True)
    registro_txt = open(destino / "registro.txt", "a", encoding="utf-8")

    def registro(texto):
        print(texto, flush=True)
        registro_txt.write(texto + "\n")
        registro_txt.flush()

    t0 = time.time()
    if not (trabajo / "camaras.json").exists():
        registro(f"Preparando imágenes en {trabajo}…")
        registro(f"{fotogrametria.preparar(origen, trabajo, a.max_fotogramas)} imágenes con su pose de ARKit")
    if not a.solo_openmvs:
        registro(f"COLMAP: {fotogrametria.colmap(trabajo, ['cmd', '/c', a.colmap], registro)}")
    registro(f"OpenMVS: {fotogrametria.openmvs(trabajo, Path(a.openmvs), registro, ['cmd', '/c', a.colmap])}")

    for f in (trabajo / "mvs").glob("texturizada*"):
        shutil.copy2(f, destino / f.name)
    shutil.copy2(trabajo / "dense" / "fused.ply", destino / "nube_densa.ply")
    for nombre in ("images.txt", "points3D.txt"):
        shutil.copy2(trabajo / "sparse" / "ajustada" / nombre, destino / f"poses_{nombre}")
    registro(f"Listo en {(time.time() - t0) / 60:.0f} min: {destino}")


if __name__ == "__main__":
    main()
