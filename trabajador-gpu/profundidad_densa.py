r"""Profundidad densa foto + LiDAR con PromptDA: une el detalle de la foto con la escala métrica del LiDAR.

    .venv-splat\Scripts\python profundidad_densa.py "C:\…\Entorno_…"

El LiDAR mide 256×192 píxeles: acierta la distancia pero borra patas, bordes y objetos finos. PromptDA recibe la foto
y ese mapa, y devuelve profundidad en metros a la resolución de la foto. Usa como prompt la lectura LiDAR real de cada
vista (el `.prof` guardado junto a la foto) y, en escaneos antiguos sin ella, la malla LiDAR proyectada.

Resultado en <escaneo>/profundidad_densa/<vista>.npy (float32, metros, a la mitad de la resolución de entrenamiento).
splat2d_local.py lo usa solo si existe esa carpeta.

Instalación (una vez). No hay que instalar sus requirements: fijan torch 2.0.1 y romperían .venv-splat, y para
inferencia solo hace falta torch:
    git clone --depth 1 https://github.com/DepthAnything/PromptDA.git %USERPROFILE%\Herramientas\PromptDA
    y guardar en Herramientas\PromptDA\checkpoints\model.ckpt el archivo de
    https://huggingface.co/depth-anything/prompt-depth-anything-vitl/resolve/main/model.ckpt
"""

from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).parent))

PROMPTDA = Path.home() / "Herramientas" / "PromptDA"
PROMPT_ALTO, PROMPT_ANCHO = 192, 256      # tamaño del mapa LiDAR de ARKit, con el que se entrenó PromptDA
PROF_MIN, PROF_MAX = 0.1, 8.0
COBERTURA_MINIMA = 0.2


def multiplo_14(n: int) -> int:
    """PromptDA divide la imagen en parches de 14 px y no redimensiona: otra medida perdería el borde."""
    return max(14, round(n / 14) * 14)


def leer_prof(ruta: Path) -> np.ndarray | None:
    """`.prof` de la app → metros, 0 = sin dato. Si trae confianza, descarta las lecturas bajas."""
    from escaner_ia.camaras import leer_profundidad

    p = leer_profundidad(ruta, {"fx": 1.0, "fy": 1.0, "cx": 0.0, "cy": 0.0})
    return None if p is None else p.mapa


def rellenar(mapa: torch.Tensor, iteraciones: int = 64) -> torch.Tensor:
    """PromptDA necesita un prompt sin huecos: cada hueco toma la media de sus vecinos válidos."""
    m = mapa[None, None].clone()
    valido = (m > 0).float()
    nucleo = torch.ones(1, 1, 3, 3, device=m.device)
    for _ in range(iteraciones):
        if valido.all():
            break
        suma = F.conv2d(m * valido, nucleo, padding=1)
        peso = F.conv2d(valido, nucleo, padding=1)
        nuevos = (valido == 0) & (peso > 0)
        m[nuevos] = (suma / peso.clamp(min=1))[nuevos]
        valido[nuevos] = 1
    return m[0, 0]


def preparar_prompt(mapa: torch.Tensor) -> torch.Tensor | None:
    """Mapa de profundidad cualquiera → 1×1×192×256 sin huecos, o `None` si casi no hay datos."""
    mapa = mapa.clone().float()
    mapa[(mapa < PROF_MIN) | (mapa > PROF_MAX)] = 0
    if (mapa > 0).float().mean() < COBERTURA_MINIMA:
        return None
    lleno = rellenar(mapa)
    if not (lleno > 0).all():
        lleno[lleno <= 0] = lleno[lleno > 0].median()
    return F.interpolate(lleno[None, None], (PROMPT_ALTO, PROMPT_ANCHO), mode="bilinear", align_corners=False)


def alinear_a_lidar(prof: torch.Tensor, lidar: torch.Tensor, iteraciones: int = 5) -> tuple[torch.Tensor, float, float]:
    """La salida de PromptDA trae un sesgo de centímetros, y las medidas las manda el LiDAR. Ajusta a·prof + b a las
    lecturas válidas con mínimos cuadrados robustos (lo que discrepa mucho, como un borde, no arrastra el ajuste)."""
    baja = F.interpolate(prof[None, None], lidar.shape, mode="bilinear", align_corners=False)[0, 0]
    valido = (lidar >= PROF_MIN) & (lidar <= PROF_MAX) & (baja > 0)
    if valido.sum() < 100:
        return prof, 1.0, 0.0
    x, y = baja[valido].double(), lidar[valido].double()
    a, b = 1.0, torch.median(y - x).item()
    for _ in range(iteraciones):
        resto = (a * x + b - y).abs()
        peso = 0.02 / resto.clamp(min=0.02)      # Huber con umbral de 2 cm
        sw, sx, sy = peso.sum(), (peso * x).sum(), (peso * y).sum()
        a = ((peso * x * y).sum() * sw - sx * sy) / ((peso * x * x).sum() * sw - sx * sx)
        b = (sy - a * sx) / sw
    a, b = float(a), float(b)
    return a * prof + b, a, b


def derivadas(prof: torch.Tensor):
    du = (prof[..., 1:-1, 2:] - prof[..., 1:-1, :-2]) / 2
    dv = (prof[..., 2:, 1:-1] - prof[..., :-2, 1:-1]) / 2
    return du, dv


def normales(prof: torch.Tensor, fx: float, fy: float) -> torch.Tensor:
    """Normales en cámara a partir de la profundidad (…×H×W → …×(H-2)×(W-2)×3). Diferenciable: se aplica la misma
    función al render y a la referencia, así que ambas quedan en la misma convención."""
    du, dv = derivadas(prof)
    z = prof[..., 1:-1, 1:-1].clamp(min=1e-3)
    n = torch.stack([-du * fx / z, -dv * fy / z, torch.ones_like(z)], -1)
    return F.normalize(n, dim=-1)


def mascara_fiable(prof: torch.Tensor, salto: float = 0.04) -> torch.Tensor:
    """Fuera de los saltos de profundidad entre objetos, donde la normal no significa nada. Tamaño (H-2)×(W-2)."""
    du, dv = derivadas(prof)
    z = prof[..., 1:-1, 1:-1].clamp(min=1e-3)
    return (prof[..., 1:-1, 1:-1] > 0) & (du.abs() / z < salto) & (dv.abs() / z < salto)


def prompts_reales(origen: Path) -> dict[str, Path]:
    """Nombre de vista → su `.prof`. Los nombres son los que da fotogrametria.preparar a las imágenes de trabajo."""
    rutas: dict[str, Path] = {}
    for sub in ("fotos_clave", "fotogramas"):
        indice = origen / sub / "camaras.json"
        if not indice.exists():
            continue
        for foto in json.loads(indice.read_text("utf-8")).get("fotos", []):
            info = foto.get("profundidad")
            if isinstance(info, dict):
                rutas[f"{sub[:3]}_{Path(foto['archivo']).stem}.jpg"] = origen / sub / info["archivo"]
    return rutas


def cargar_modelo(dev: torch.device):
    """PromptDA desde su carpeta. Solo usa huggingface_hub para descargar, y aquí el modelo ya está en disco."""
    import types

    sys.path.insert(0, str(PROMPTDA))
    if "huggingface_hub" not in sys.modules:
        try:
            import huggingface_hub  # noqa: F401
        except ImportError:
            sys.modules["huggingface_hub"] = types.SimpleNamespace(hf_hub_download=None)
    from promptda.promptda import PromptDA

    ruta = PROMPTDA / "checkpoints" / "model.ckpt"
    if not ruta.exists():
        raise SystemExit(f"Falta el modelo: {ruta}")
    return PromptDA.from_pretrained(str(ruta)).to(dev).eval()


@torch.no_grad()
def main():
    from splat2d_local import HERRAMIENTAS, cargar_vistas, leer_ply, profundidad_lidar

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    p.add_argument("--lado", type=int, default=1280, help="lado mayor de las imágenes, como en splat2d_local")
    p.add_argument("--limite", type=int, default=0, help="procesar solo las N primeras vistas (para probar)")
    a = p.parse_args()

    origen = Path(a.carpeta).resolve()
    trabajo = HERRAMIENTAS / "trabajos" / origen.name
    salida = origen / "profundidad_densa"
    salida.mkdir(exist_ok=True)
    T = np.array(json.loads((origen / "fotogrametria" / "colmap_a_lidar.json").read_text())["matriz"])
    vistas = cargar_vistas(trabajo, T, a.lado)
    if a.limite:
        vistas = vistas[:a.limite]

    dev = torch.device("cuda")
    modelo = cargar_modelo(dev)
    lidar = torch.from_numpy(leer_ply(origen / "malla_color.ply")[0]).to(dev)
    reales = prompts_reales(origen)

    usadas = {"lidar": 0, "malla": 0, "sin_prompt": 0}
    ajustes: list[tuple[float, float]] = []
    t0 = time.time()
    for i, v in enumerate(vistas):
        destino = salida / (Path(v["nombre"]).stem + ".npy")
        if destino.exists():
            continue
        prompt, fuente, referencia = None, "sin_prompt", None
        ruta = reales.get(v["nombre"])
        if ruta is not None and (mapa := leer_prof(ruta)) is not None:
            referencia = torch.from_numpy(mapa.copy()).to(dev)
            prompt = preparar_prompt(referencia)
            fuente = "lidar"
        if prompt is None:
            proyectada = profundidad_lidar(lidar, v["w2c"], v["K"], v["ancho"], v["alto"], red=2)
            referencia = proyectada
            prompt = preparar_prompt(proyectada)
            fuente = "malla"
        if prompt is None:
            usadas["sin_prompt"] += 1
            continue
        usadas[fuente] += 1

        img = v["img"].to(dev).float().permute(2, 0, 1)[None] / 255
        alto, ancho = multiplo_14(v["alto"]), multiplo_14(v["ancho"])
        img = F.interpolate(img, (alto, ancho), mode="bilinear", align_corners=False)
        with torch.autocast("cuda", dtype=torch.float16):
            prof = modelo.predict(img, prompt).float()
        prof = F.interpolate(prof.reshape(1, 1, *prof.shape[-2:]), (v["alto"] // 2, v["ancho"] // 2),
                             mode="bilinear", align_corners=False)[0, 0]
        prof, escala, desfase = alinear_a_lidar(prof, referencia)
        ajustes.append((escala, desfase))
        np.save(destino, prof.cpu().numpy().astype(np.float32))
        if i % 20 == 0:
            print(f"Profundidad densa: {i + 1}/{len(vistas)}", flush=True)
    print(f"Profundidad densa lista en {time.time() - t0:.0f} s: {usadas}", flush=True)
    if ajustes:
        e, d = np.median(np.array(ajustes), axis=0)
        print(f"Ajuste mediano al LiDAR: escala {e:.4f}, desfase {d * 1000:.1f} mm", flush=True)


if __name__ == "__main__":
    main()
