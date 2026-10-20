"""Etapa F: Gaussian Splatting con gsplat (Apache 2.0) — vista fotorrealista de la escena.

A diferencia del método original, no hace falta COLMAP: las poses de cada foto vienen de ARKit y los
puntos iniciales de la malla LiDAR, así que el resultado sale a escala real y alineado con la malla
(y con los objetos de escena.json).

Vistas: fotos de 12 MP (`fotos_clave/`) + fotogramas extra (`fotogramas/`), ambos con su `camaras.json`.
Sin fotos (escaneos antiguos) se entrena con imágenes sintéticas de la malla: sirve para probar el proceso,
no para obtener calidad.

Salida: `.splat` (formato compacto de 32 bytes por gaussiana: posición, escala, color RGBA y rotación),
que abren Spark, SuperSplat y la mayoría de visores web.
"""

from __future__ import annotations

import math
import random
import time
from dataclasses import dataclass
from pathlib import Path

import numpy as np

from . import camaras, ply, render

C0 = 0.28209479177387814   # coeficiente del armónico esférico de grado 0


@dataclass
class ParametrosSplats:
    pasos: int = 7000
    lado_max: int = 1280              # resolución de entrenamiento (lado mayor)
    max_puntos_iniciales: int = 300_000
    refinar_hasta: int = 5000         # densificación (dividir/clonar/podar) hasta este paso
    max_gaussianas: int = 1_000_000   # al exportar: ~32 MB en .splat
    opacidad_minima: float = 0.005


def cargar_vistas(carpeta: Path, base: ply.Malla, lado_max: int, avisos: list[str]):
    vistas = []
    for sub in ("fotos_clave", "fotogramas"):
        if (carpeta / sub / "camaras.json").exists():
            v, av = camaras.cargar_fotos(carpeta / sub, lado_max=lado_max)
            vistas += v
            avisos += av
    if vistas:
        return vistas, "fotos"
    avisos.append("Gaussian Splatting sin fotos: entrenado con imágenes sintéticas de la malla (solo para probar).")
    virtuales = camaras.camaras_virtuales(base.posiciones)
    return render.fotografiar(base.posiciones, base.colores, base.caras, virtuales), "sinteticas"


def generar(carpeta: Path, base: ply.Malla, destino: Path, avance, avisos: list[str],
            p: ParametrosSplats | None = None) -> dict:
    p = p or ParametrosSplats()
    vistas, tipo = cargar_vistas(carpeta, base, p.lado_max, avisos)
    if len(vistas) < 10:
        raise ValueError(f"Hacen falta al menos 10 vistas y hay {len(vistas)}")
    t0 = time.time()
    gaussianas = entrenar(vistas, base, p, avance)
    destino.parent.mkdir(parents=True, exist_ok=True)
    n = escribir_splat(destino, gaussianas, p)
    return {"vistas": len(vistas), "tipo": tipo, "gaussianas": n, "pasos": p.pasos,
            "segundos": round(time.time() - t0, 1), "archivo": destino.name}


# ------------------------------------------------------------------------------------------- entrenamiento

def _ssim(x, y):
    """SSIM con ventana gaussiana 11×11 (x, y: 1×3×H×W en 0–1)."""
    import torch
    import torch.nn.functional as F

    g = torch.exp(-((torch.arange(11, device=x.device) - 5.0) ** 2) / (2 * 1.5 ** 2))
    g = (g / g.sum())
    ventana = (g[:, None] @ g[None, :]).expand(3, 1, 11, 11).contiguous()
    filtro = lambda t: F.conv2d(t, ventana, padding=5, groups=3)
    mx, my = filtro(x), filtro(y)
    sxx, syy, sxy = filtro(x * x) - mx ** 2, filtro(y * y) - my ** 2, filtro(x * y) - mx * my
    c1, c2 = 0.01 ** 2, 0.03 ** 2
    return (((2 * mx * my + c1) * (2 * sxy + c2)) / ((mx ** 2 + my ** 2 + c1) * (sxx + syy + c2))).mean()


def entrenar(vistas, base: ply.Malla, p: ParametrosSplats, avance) -> dict:
    import torch
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy
    from scipy.spatial import cKDTree

    dispositivo = "cuda"
    torch.manual_seed(0)
    random.seed(0)

    imagenes = [torch.from_numpy(np.ascontiguousarray(v.imagen)).to(dispositivo) for v in vistas]
    ks = torch.tensor([[[v.fx, 0, v.cx], [0, v.fy, v.cy], [0, 0, 1]] for v in vistas],
                      dtype=torch.float32, device=dispositivo)
    viewmats = torch.tensor(np.stack([v.mundo_a_cam for v in vistas]), dtype=torch.float32, device=dispositivo)

    # Escala de la escena: lo lejos que están las cámaras de su centro (como el entrenador de gsplat).
    centros = np.stack([v.posicion for v in vistas])
    escala = float(max(1.1 * np.linalg.norm(centros - centros.mean(0), axis=1).max(), 1.0))

    # Puntos iniciales: vértices de la malla LiDAR con su color.
    puntos, colores = base.posiciones.astype(np.float32), base.colores.astype(np.float32) / 255
    if len(puntos) > p.max_puntos_iniciales:
        elegidos = np.random.default_rng(0).choice(len(puntos), p.max_puntos_iniciales, replace=False)
        puntos, colores = puntos[elegidos], colores[elegidos]
    distancias, _ = cKDTree(puntos).query(puntos, k=4)
    radio = np.clip(distancias[:, 1:].mean(1), 1e-3, 0.1).astype(np.float32)   # gsplat trabaja en float32

    n = len(puntos)
    cuat = torch.randn(n, 4)
    params = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(torch.from_numpy(puntos)),
        "scales": torch.nn.Parameter(torch.log(torch.from_numpy(radio)).unsqueeze(1).repeat(1, 3)),
        "quats": torch.nn.Parameter(cuat / cuat.norm(dim=1, keepdim=True)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1))),
        "sh0": torch.nn.Parameter(((torch.from_numpy(colores) - 0.5) / C0).unsqueeze(1)),
    }).to(dispositivo)
    tasas = {"means": 1.6e-4 * escala, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3}
    optimizadores = {nombre: torch.optim.Adam([{"params": params[nombre], "lr": tasa, "name": nombre}], eps=1e-15)
                     for nombre, tasa in tasas.items()}
    planificador = torch.optim.lr_scheduler.ExponentialLR(optimizadores["means"], gamma=0.01 ** (1 / p.pasos))

    estrategia = DefaultStrategy(refine_stop_iter=p.refinar_hasta, reset_every=3000, verbose=False)
    estrategia.check_sanity(params, optimizadores)
    estado = estrategia.initialize_state(scene_scale=escala)

    orden = list(range(len(vistas)))
    for paso in range(p.pasos):
        if paso % len(orden) == 0:
            random.shuffle(orden)
        i = orden[paso % len(orden)]
        objetivo = imagenes[i].float() / 255
        alto, ancho = objetivo.shape[:2]

        render_, _, info = rasterization(
            means=params["means"], quats=params["quats"], scales=torch.exp(params["scales"]),
            opacities=torch.sigmoid(params["opacities"]), colors=params["sh0"],
            viewmats=viewmats[i:i + 1], Ks=ks[i:i + 1], width=ancho, height=alto, sh_degree=0, packed=False)
        estrategia.step_pre_backward(params, optimizadores, estado, paso, info)

        prediccion = render_[0]
        l1 = (prediccion - objetivo).abs().mean()
        ssim = _ssim(prediccion.permute(2, 0, 1)[None], objetivo.permute(2, 0, 1)[None])
        perdida = 0.8 * l1 + 0.2 * (1 - ssim)
        perdida.backward()

        for opt in optimizadores.values():
            opt.step()
            opt.zero_grad(set_to_none=True)
        planificador.step()
        estrategia.step_post_backward(params, optimizadores, estado, paso, info, packed=False)

        if paso % 100 == 0:
            avance(paso / p.pasos)

    with torch.no_grad():
        return {
            "means": params["means"].cpu().numpy(),
            "scales": torch.exp(params["scales"]).cpu().numpy(),
            "quats": torch.nn.functional.normalize(params["quats"], dim=1).cpu().numpy(),
            "opacities": torch.sigmoid(params["opacities"]).cpu().numpy(),
            "colores": np.clip(0.5 + C0 * params["sh0"][:, 0].cpu().numpy(), 0, 1),
        }


# ------------------------------------------------------------------------------------------- exportación

def escribir_splat(ruta: Path, g: dict, p: ParametrosSplats) -> int:
    """Formato `.splat`: por gaussiana, posición (3×f32), escala (3×f32), RGBA (4×u8) y rotación w,x,y,z (4×u8).
    Ordenadas de más a menos visibles (opacidad × volumen), que es lo que esperan los visores."""
    opacidad = g["opacities"]
    importancia = opacidad * np.prod(g["scales"], axis=1)
    orden = np.argsort(-importancia)
    orden = orden[opacidad[orden] >= p.opacidad_minima][:p.max_gaussianas]

    n = len(orden)
    registro = np.dtype([("pos", "<f4", 3), ("escala", "<f4", 3), ("rgba", "u1", 4), ("rot", "u1", 4)])
    datos = np.empty(n, registro)
    datos["pos"] = g["means"][orden]
    datos["escala"] = g["scales"][orden]
    rgba = np.concatenate([g["colores"][orden], opacidad[orden, None]], axis=1)
    datos["rgba"] = np.clip(rgba * 255, 0, 255).astype(np.uint8)
    datos["rot"] = np.clip(g["quats"][orden] * 128 + 128, 0, 255).astype(np.uint8)
    ruta.write_bytes(datos.tobytes())
    return n
