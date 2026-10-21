r"""Procesa un escaneo de principio a fin en este PC: un solo comando.

    procesar_escaneo.bat "C:\…\Entorno_…"

1. Fotogrametría COLMAP con las poses de ARKit (puntos, ajuste de haces a escala real, nube densa).
2. Alineación automática con la malla LiDAR.
3. Gaussian Splatting 2D guiado por la profundidad LiDAR y extracción de la malla.
4. Suavizado y ajuste de suelo, techo y paredes a planos.
5. Versión ligera para el iPhone.

Cada paso se salta si su resultado ya existe, así que si algo falla basta con volver a lanzarlo.
Resultado en <escaneo>/splat2d/: malla_2dgs_plana.ply (completa) y malla_movil.ply (ligera), además de splats.ply.
"""

from __future__ import annotations

import argparse
import os
import subprocess
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from escaner_ia import fotogrametria  # noqa: E402

AQUI = Path(__file__).parent
HERRAMIENTAS = Path.home() / "Herramientas"
COLMAP = ["cmd", "/c", str(HERRAMIENTAS / "colmap" / "COLMAP.bat")]


def ejecutar(args: list[str], registro):
    """Ejecuta un paso en otro proceso y pasa cada línea de su salida a `registro` (para el progreso)."""
    proceso = subprocess.Popen(args, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, text=True,
                               encoding="utf-8", errors="replace", env={**os.environ, "PYTHONIOENCODING": "utf-8"})
    ultimas = []
    for linea in proceso.stdout:
        linea = linea.rstrip()
        ultimas = (ultimas + [linea])[-20:]
        registro(linea)
    if proceso.wait() != 0:
        raise RuntimeError(f"{Path(args[1]).name} falló:\n" + "\n".join(ultimas))


def malla_movil(entrada: Path, salida: Path, triangulos: int, registro):
    """Simplificación por error cuadrático conservando el color por vértice (el más cercano)."""
    import fast_simplification
    import numpy as np
    from scipy.spatial import cKDTree

    from splat2d_local import guardar_malla_ply
    from suavizar_malla import leer

    v, c, f = leer(entrada)
    reduccion = max(0.0, 1 - triangulos / len(f))
    v2, f2 = fast_simplification.simplify(v.astype(np.float32), f.astype(np.int32), target_reduction=reduccion)
    c2 = c[cKDTree(v).query(v2)[1]]
    guardar_malla_ply(salida, v2.astype(np.float32), c2, f2.astype(np.int32))
    registro(f"Malla para el iPhone: {len(f2)} triángulos ({salida.stat().st_size / 1e6:.0f} MB)")


def procesar(carpeta: Path, max_fotogramas: int = 160, pasos: int = 20_000, triangulos_movil: int = 1_500_000,
             registro=print) -> Path:
    origen = carpeta.resolve()
    trabajo = HERRAMIENTAS / "trabajos" / origen.name
    destino = origen / "splat2d"
    destino.mkdir(exist_ok=True)
    t0 = time.time()

    if not (trabajo / "camaras.json").exists():
        registro("1/5 Preparando imágenes…")
        registro(f"    {fotogrametria.preparar(origen, trabajo, max_fotogramas)} imágenes con su pose de ARKit")
    if not (trabajo / "dense" / "fused.ply").exists():
        registro("1/5 Fotogrametría COLMAP (la parte más larga: profundidad densa en la GPU)…")
        registro(f"    {fotogrametria.colmap(trabajo, COLMAP, registro)}")
    else:
        registro("1/5 Fotogrametría: ya hecha")

    # 2-3: alineación, entrenamiento y malla (splat2d_local se encarga de la alineación si falta).
    if not (destino / "malla_2dgs.ply").exists():
        registro("2-3/5 Alineación con el LiDAR, Gaussian Splatting 2D y malla…")
        args = [sys.executable, str(AQUI / "splat2d_local.py"), str(origen), "--pasos", str(pasos)]
        if (destino / "splats.pt").exists():
            args.append("--solo-malla")
        ejecutar(args, registro)
    else:
        registro("2-3/5 Malla 2DGS: ya hecha")

    plana = destino / "malla_2dgs_plana.ply"
    if not plana.exists():
        registro("4/5 Suavizado y planos…")
        ejecutar([sys.executable, str(AQUI / "aplanar_malla.py"), str(destino / "malla_2dgs.ply")], registro)

    movil = destino / "malla_movil.ply"
    if not movil.exists():
        registro("5/5 Versión ligera…")
        malla_movil(plana, movil, triangulos_movil, registro)

    registro(f"Listo en {(time.time() - t0) / 60:.0f} min → {destino}")
    return destino


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    p.add_argument("--max-fotogramas", type=int, default=160)
    p.add_argument("--pasos", type=int, default=20_000)
    p.add_argument("--triangulos-movil", type=int, default=1_500_000)
    a = p.parse_args()
    procesar(Path(a.carpeta), a.max_fotogramas, a.pasos, a.triangulos_movil)


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    main()
