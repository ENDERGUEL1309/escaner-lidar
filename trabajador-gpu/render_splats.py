r"""Render fotorrealista de todo el entorno: Gaussian Splatting 3D entrenado en este PC para verse, no para medir.

    .venv-splat\Scripts\python render_splats.py "C:\…\Entorno_…"

splat2d_local.py entrena discos planos para sacar una malla; aquí cada gaussiana es un elipsoide libre y la pérdida
es sobre todo la foto, así que reflejos, texturas finas y bordes salen como en la cámara. Usa las poses ajustadas por
COLMAP, corrige la exposición de cada foto y, si existe, la profundidad densa (profundidad_densa.py) frena las
gaussianas sueltas que flotan delante de las paredes.

Una de cada 8 vistas no se usa para entrenar: con ellas se mide la calidad en fotos que el modelo no ha visto.
Resultado en <escaneo>/splats_render/: escena.splat (lo que abre la app), calidad.json y comparacion.jpg.
"""

from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image

sys.path.insert(0, str(Path(__file__).parent))
from escaner_ia.splats import ParametrosSplats, escribir_splat  # noqa: E402
from splat2d_local import (HERRAMIENTAS, cargar_vistas, leer_ply, profundidad_lidar,  # noqa: E402
                           rgb_a_sh, ssim)

C0 = 0.28209479177387814


def psnr(a: torch.Tensor, b: torch.Tensor) -> float:
    return float(-10 * torch.log10(((a - b) ** 2).mean().clamp(min=1e-10)))


def con_ganancia(pred: torch.Tensor, gt: torch.Tensor) -> torch.Tensor:
    """La exposición de una vista reservada no se aprendió: se ajusta una ganancia por canal antes de comparar."""
    g = (pred * gt).sum((0, 1)) / (pred * pred).sum((0, 1)).clamp(min=1e-8)
    return (pred * g).clamp(0, 1)


def main():
    from gsplat import rasterization
    from gsplat.strategy import DefaultStrategy

    p = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    p.add_argument("carpeta")
    p.add_argument("--pasos", type=int, default=30_000)
    p.add_argument("--lado", type=int, default=1280)
    p.add_argument("--max-gaussianas", type=int, default=1_500_000, help="en el .splat (≈48 MB con 1,5 millones)")
    p.add_argument("--tope-entrenamiento", type=int, default=3_500_000, help="no crecer más allá (12 GB de GPU)")
    p.add_argument("--peso-profundidad", type=float, default=0.05)
    p.add_argument("--densa", default="profundidad_densa")
    p.add_argument("--salida", default="splats_render")
    a = p.parse_args()

    origen = Path(a.carpeta).resolve()
    trabajo = HERRAMIENTAS / "trabajos" / origen.name
    destino = origen / a.salida
    destino.mkdir(exist_ok=True)
    dev = torch.device("cuda")
    torch.manual_seed(0)

    T = np.array(json.loads((origen / "fotogrametria" / "colmap_a_lidar.json").read_text())["matriz"])
    vistas = cargar_vistas(trabajo, T, a.lado)
    lidar_xyz, lidar_rgb = leer_ply(origen / "malla_color.ply")
    lidar = torch.from_numpy(lidar_xyz).to(dev)
    for v in vistas:
        densa = origen / a.densa / (Path(v["nombre"]).stem + ".npy")
        v["prof"] = (torch.from_numpy(np.load(densa)).float() if densa.exists()
                     else profundidad_lidar(lidar, v["w2c"], v["K"], v["ancho"], v["alto"]).cpu())
    prueba = set(range(0, len(vistas), 8))
    entreno = [i for i in range(len(vistas)) if i not in prueba]
    print(f"{len(vistas)} vistas: {len(entreno)} para entrenar y {len(prueba)} reservadas para medir", flush=True)

    imagenes = [v["img"].to(dev) for v in vistas]
    Ks = torch.from_numpy(np.stack([v["K"] for v in vistas])).to(dev)
    w2cs = torch.from_numpy(np.stack([v["w2c"] for v in vistas])).to(dev)
    centros = np.array([np.linalg.inv(v["w2c"])[:3, 3] for v in vistas])
    # Escala de la escena: lo que abarcan las cámaras o, si se grabó girando desde el centro (cámaras a pocos cm),
    # lo que abarca la habitación medida por el LiDAR. Con solo las cámaras salía 0,3 m y todo lo más lejano se perdía.
    escena = float(max(1.1 * np.linalg.norm(centros - centros.mean(0), axis=1).max(),
                       np.percentile(np.linalg.norm(lidar_xyz[::50] - centros.mean(0), axis=1), 90), 1.0))

    # Inicio: nube densa de COLMAP (detalle donde hay textura) + malla LiDAR (paredes lisas), como splat2d_local.
    from escaner_ia.fotogrametria import nube_inicial
    densa_xyz, densa_rgb = nube_inicial(trabajo)
    densa_xyz = densa_xyz @ T[:3, :3].T.astype(np.float32) + T[:3, 3].astype(np.float32)
    rng = np.random.default_rng(0)
    i1 = rng.choice(len(densa_xyz), min(500_000, len(densa_xyz)), replace=False)
    i2 = rng.choice(len(lidar_xyz), min(800_000 - len(i1), len(lidar_xyz)), replace=False)
    xyz = torch.from_numpy(np.concatenate([densa_xyz[i1], lidar_xyz[i2]])).to(dev)
    rgb = torch.from_numpy(np.concatenate([densa_rgb[i1], lidar_rgb[i2]])).to(dev)
    d = torch.empty(len(xyz), device=dev)
    for i in range(0, len(xyz), 512):
        d[i:i + 512] = torch.cdist(xyz[i:i + 512], xyz).topk(4, largest=False).values[:, 1:].mean(1)
    n = len(xyz)
    splats = torch.nn.ParameterDict({
        "means": torch.nn.Parameter(xyz),
        "scales": torch.nn.Parameter(torch.log(d.clamp(1e-3, 0.05))[:, None].repeat(1, 3)),
        "quats": torch.nn.Parameter(F.normalize(torch.randn(n, 4, device=dev), dim=-1)),
        "opacities": torch.nn.Parameter(torch.logit(torch.full((n,), 0.1, device=dev))),
        "sh0": torch.nn.Parameter(rgb_a_sh(rgb)[:, None, :]),
    })
    tasas = {"means": 1.6e-4 * escena, "scales": 5e-3, "quats": 1e-3, "opacities": 5e-2, "sh0": 2.5e-3}
    optim = {k: torch.optim.Adam([{"params": splats[k], "lr": tasas[k], "name": k}], eps=1e-15) for k in splats}
    decaimiento = torch.optim.lr_scheduler.ExponentialLR(optim["means"], gamma=0.01 ** (1 / a.pasos))
    expo = torch.nn.Parameter(torch.zeros(len(vistas), 3, device=dev))
    optim_expo = torch.optim.Adam([expo], lr=1e-3)

    estrategia = DefaultStrategy(refine_start_iter=500, refine_stop_iter=a.pasos // 2, reset_every=3000,
                                 refine_every=100, grow_grad2d=0.0002, prune_opa=0.005, verbose=False)
    estrategia.check_sanity(splats, optim)
    estado = estrategia.initialize_state(scene_scale=escena)

    def dibujar(i, con_profundidad=False):
        render, alfa, info = rasterization(
            means=splats["means"], quats=splats["quats"], scales=torch.exp(splats["scales"]),
            opacities=torch.sigmoid(splats["opacities"]), colors=splats["sh0"], viewmats=w2cs[i:i + 1],
            Ks=Ks[i:i + 1], width=vistas[i]["ancho"], height=vistas[i]["alto"], sh_degree=0, packed=False,
            near_plane=0.05, far_plane=20.0, rasterize_mode="antialiased",
            render_mode="RGB+ED" if con_profundidad else "RGB")
        return render[0], alfa[0], info

    print(f"Render: {a.pasos} pasos con {n} gaussianas iniciales…", flush=True)
    t0 = time.time()
    orden = np.random.default_rng(1).permutation(np.tile(entreno, a.pasos // len(entreno) + 1))
    for paso in range(a.pasos):
        i = int(orden[paso])
        gt = imagenes[i].float() / 255
        render, alfa, info = dibujar(i, con_profundidad=True)
        estrategia.step_pre_backward(splats, optim, estado, paso, info)
        color = render[..., :3] * torch.exp(expo[i])
        perdida = 0.8 * (color - gt).abs().mean() + 0.2 * (1 - ssim(color.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))
        perdida = perdida + 0.01 * (expo[i] ** 2).sum()
        ref = vistas[i]["prof"].to(dev)
        red = round(vistas[i]["alto"] / ref.shape[0])
        prof = F.avg_pool2d(render[..., 3:4].permute(2, 0, 1)[None], red)[0, 0]
        hh, ww = min(ref.shape[0], prof.shape[0]), min(ref.shape[1], prof.shape[1])
        ref, prof = ref[:hh, :ww], prof[:hh, :ww]
        valida = ref > 0
        if valida.any():
            perdida = perdida + a.peso_profundidad * ((prof - ref).abs() / ref)[valida].clamp(max=0.5).mean()
        perdida.backward()
        for prm in list(splats.values()) + [expo]:
            if prm.grad is not None:
                prm.grad.nan_to_num_(0.0, 0.0, 0.0)
        for o in list(optim.values()) + [optim_expo]:
            o.step()
            o.zero_grad(set_to_none=True)
        decaimiento.step()
        # Con 12 GB de GPU no se puede crecer sin límite: pasado el tope solo se poda.
        if len(splats["means"]) < a.tope_entrenamiento or paso % 100 != 0:
            estrategia.step_post_backward(splats, optim, estado, paso, info, packed=False)
        if paso % 1000 == 0 or paso == a.pasos - 1:
            print(f"  render {paso:6d}  L1 {(color - gt).abs().mean().item():.4f}  gaussianas {len(splats['means'])}  "
                  f"({(time.time() - t0) / 60:.1f} min)", flush=True)

    calidad = {"pasos": a.pasos, "minutos": round((time.time() - t0) / 60, 1), "vistas_reservadas": len(prueba)}
    filas, medidas = [], []
    with torch.no_grad():
        for k, i in enumerate(sorted(prueba)):
            render, _, _ = dibujar(i)
            gt = imagenes[i].float() / 255
            pred = con_ganancia(render.clamp(0, 1), gt)
            medidas.append((psnr(pred, gt), float(ssim(pred.permute(2, 0, 1)[None], gt.permute(2, 0, 1)[None]))))
            if k in (1, len(prueba) // 2, len(prueba) - 2):
                filas.append(np.concatenate([imagenes[i].cpu().numpy(), (pred * 255).byte().cpu().numpy()], 1))
        calidad["psnr_reservadas"] = round(float(np.mean([m[0] for m in medidas])), 2)
        calidad["ssim_reservadas"] = round(float(np.mean([m[1] for m in medidas])), 4)
        g = {
            "means": splats["means"].cpu().numpy(),
            "scales": torch.exp(splats["scales"]).cpu().numpy(),
            "quats": F.normalize(splats["quats"], dim=1).cpu().numpy(),
            "opacities": torch.sigmoid(splats["opacities"]).cpu().numpy(),
            "colores": np.clip(0.5 + C0 * splats["sh0"][:, 0].cpu().numpy(), 0, 1),
        }
    calidad["gaussianas"] = escribir_splat(destino / "escena.splat", g,
                                           ParametrosSplats(max_gaussianas=a.max_gaussianas, opacidad_minima=0.01))
    (destino / "calidad.json").write_text(json.dumps(calidad, indent=2))
    if filas:
        Image.fromarray(np.concatenate(filas, 0)).save(destino / "comparacion.jpg", quality=90)
    print(f"Render listo: {calidad}", flush=True)


if __name__ == "__main__":
    main()
