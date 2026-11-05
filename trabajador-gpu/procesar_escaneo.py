r"""Procesa un escaneo de principio a fin en este PC: un solo comando.

    procesar_escaneo.bat "C:\…\Entorno_…"

1. Fotogrametría COLMAP con las poses de ARKit (puntos, ajuste de haces a escala real, nube densa).
2. Alineación automática con la malla LiDAR.
3. Gaussian Splatting 2D guiado por la profundidad LiDAR y extracción de la malla.
4. Suavizado y ajuste de suelo, techo y paredes a planos.
5. Versión ligera para el iPhone.
6. Modelado verificado: los muebles reconocidos pasan a ser modelos limpios ajustados contra las fotos.

Cada paso se salta si su resultado ya existe, así que si algo falla basta con volver a lanzarlo.
Resultado en <escaneo>/splat2d/: malla_2dgs_plana.ply (completa), malla_movil.ply (ligera), escena_modelada.ply
(con los muebles modelados) y splats.ply.
"""

from __future__ import annotations

import argparse
import hashlib
import os
import shutil
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


def huella(origen: Path, max_fotogramas: int) -> str:
    """Identifica las entradas de la fotogrametría: mismas fotos y poses (camaras.json) y mismo tope de fotogramas."""
    h = hashlib.sha256(str(max_fotogramas).encode())
    for sub in ("fotos_clave", "fotogramas"):
        ruta = origen / sub / "camaras.json"
        if ruta.exists():
            h.update(sub.encode())
            h.update(ruta.read_bytes())
    return h.hexdigest()


def reutilizar_fotogrametria(origen: Path, trabajo: Path, max_fotogramas: int, registro) -> bool:
    """Al reenviar un escaneo ya procesado, el trabajo tiene otro nombre pero las mismas fotos: se copia la
    fotogrametría hecha (imágenes, puntos y ajuste de haces; la nube densa solo si existe) en vez de repetirla."""
    if trabajo.exists():
        return False
    buscada = huella(origen, max_fotogramas)
    for previo in sorted((HERRAMIENTAS / "trabajos").glob("*/huella.txt"), key=lambda p: -p.stat().st_mtime):
        anterior = previo.parent
        if previo.read_text().strip() != buscada or not (anterior / "sparse" / "ajustada" / "points3D.txt").exists():
            continue
        trabajo.mkdir(parents=True)
        for nombre in ("camaras.json", "huella.txt", "database.db"):
            if (anterior / nombre).exists():
                shutil.copy2(anterior / nombre, trabajo / nombre)
        shutil.copytree(anterior / "imagenes", trabajo / "imagenes")
        shutil.copytree(anterior / "sparse", trabajo / "sparse")
        if (anterior / "dense" / "fused.ply").exists():
            (trabajo / "dense").mkdir()
            shutil.copy2(anterior / "dense" / "fused.ply", trabajo / "dense" / "fused.ply")
        registro(f"1/6 Fotogrametría reutilizada de {anterior.name} (mismas fotos y poses)")
        return True
    return False


def procesar(carpeta: Path, max_fotogramas: int = 160, pasos: int = 20_000, triangulos_movil: int = 1_500_000,
             registro=print, densa: bool = False) -> Path:
    origen = carpeta.resolve()
    trabajo = HERRAMIENTAS / "trabajos" / origen.name
    destino = origen / "splat2d"
    destino.mkdir(exist_ok=True)
    t0 = time.time()

    # Habitación grabada en la app: trae fotogramas con LiDAR pero no malla; se funde aquí.
    if not (origen / "malla_color.ply").exists() and (origen / "fotogramas" / "camaras.json").exists():
        registro("0/6 Fusión LiDAR de la grabación…")
        ejecutar([sys.executable, str(AQUI / "fusionar_lidar.py"), str(origen)], registro)

    reutilizar_fotogrametria(origen, trabajo, max_fotogramas, registro)
    if not (trabajo / "camaras.json").exists():
        registro("1/6 Preparando imágenes…")
        registro(f"    {fotogrametria.preparar(origen, trabajo, max_fotogramas)} imágenes con su pose de ARKit")
        (trabajo / "huella.txt").write_text(huella(origen, max_fotogramas))
    if not (trabajo / "sparse" / "ajustada" / "points3D.txt").exists():
        registro("1/6 Fotogrametría COLMAP (puntos y ajuste de haces con las poses de ARKit)…")
        registro(f"    {fotogrametria.colmap(trabajo, COLMAP, registro, densa=densa)}")
    else:
        registro("1/6 Fotogrametría: ya hecha")

    # 2-3: alineación, entrenamiento y malla (splat2d_local se encarga de la alineación si falta).
    if not (destino / "malla_2dgs.ply").exists():
        registro("2-3/6 Alineación con el LiDAR, Gaussian Splatting 2D y malla…")
        args = [sys.executable, str(AQUI / "splat2d_local.py"), str(origen), "--pasos", str(pasos)]
        if (destino / "splats.pt").exists():
            args.append("--solo-malla")
        ejecutar(args, registro)
    else:
        registro("2-3/6 Malla 2DGS: ya hecha")

    plana = destino / "malla_2dgs_plana.ply"
    if not plana.exists():
        registro("4/6 Suavizado y planos…")
        ejecutar([sys.executable, str(AQUI / "aplanar_malla.py"), str(destino / "malla_2dgs.ply")], registro)

    movil = destino / "malla_movil.ply"
    if not movil.exists():
        registro("5/6 Versión ligera…")
        malla_movil(plana, movil, triangulos_movil, registro)

    # 6: muebles reconocidos (cama, escritorio, silla con modelo de IA) sustituidos por modelos limpios, ajustados
    # y verificados contra las fotos. Si algo falla aquí, la malla escaneada sigue siendo un resultado válido.
    if not (destino / "escena_modelada.ply").exists():
        registro("6/6 Modelado verificado con fotos…")
        try:
            ejecutar([sys.executable, str(AQUI / "modelado_verificado.py"), str(origen)], registro)
        except RuntimeError as error:
            registro(f"    sin modelado: {str(error).splitlines()[-1]}")

    # 7: render fotorrealista de todo el entorno (Gaussian Splatting 3D). Si falla, la malla sigue siendo válida.
    if not (origen / "splats_render" / "escena.splat").exists():
        registro("7/7 Render fotorrealista…")
        try:
            ejecutar([sys.executable, str(AQUI / "render_splats.py"), str(origen)], registro)
        except RuntimeError as error:
            registro(f"    sin render: {str(error).splitlines()[-1]}")

    registro(f"Listo en {(time.time() - t0) / 60:.0f} min → {destino}")
    return destino


def main():
    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    p.add_argument("--max-fotogramas", type=int, default=160)
    p.add_argument("--pasos", type=int, default=20_000)
    p.add_argument("--triangulos-movil", type=int, default=1_500_000)
    p.add_argument("--densa", action="store_true", help="calcular también la nube densa de COLMAP (20-50 min más)")
    a = p.parse_args()
    procesar(Path(a.carpeta), a.max_fotogramas, a.pasos, a.triangulos_movil, densa=a.densa)


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    main()
