"""Pruebas de la parte de profundidad_densa.py que no necesita el modelo.

    .venv-splat\\Scripts\\python pruebas\\prueba_densa.py
"""

import json
import struct
import sys
import tempfile
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
import profundidad_densa as pd  # noqa: E402


def prueba_multiplo_14():
    assert pd.multiplo_14(1280) == 1274
    assert pd.multiplo_14(960) == 966
    assert pd.multiplo_14(3) == 14


def prueba_prompt_sin_huecos():
    m = torch.full((96, 128), 2.0)
    m[40:50, 60:70] = 0                # hueco
    m[0:5, 0:5] = 50.0                 # lectura imposible
    prompt = pd.preparar_prompt(m)
    assert prompt.shape == (1, 1, 192, 256)
    assert (prompt > 0).all()
    assert (prompt - 2).abs().max() < 1e-3


def prueba_prompt_vacio():
    assert pd.preparar_prompt(torch.zeros(96, 128)) is None


def prueba_normales_de_un_plano():
    # Pared frontal: la normal apunta a la cámara.
    n = pd.normales(torch.full((20, 30), 2.0), 500.0, 500.0)
    assert (n - torch.tensor([0.0, 0, 1])).abs().max() < 1e-6
    # Plano inclinado z = 2 + 0,001 u: gira hacia −x y la función es diferenciable.
    u = torch.arange(30.0)[None].repeat(20, 1)
    prof = (2 + 0.001 * u).requires_grad_()
    n = pd.normales(prof, 500.0, 500.0)
    assert (n[..., 0] < -0.1).all() and n[..., 1].abs().max() < 1e-6
    n.sum().backward()
    assert prof.grad is not None


def prueba_mascara_excluye_saltos():
    prof = torch.full((20, 30), 2.0)
    prof[:, 15:] = 3.0
    m = pd.mascara_fiable(prof)
    assert m.shape == (18, 28)
    assert not m[:, 13:15].any()
    assert m[:, :12].all() and m[:, 16:].all()


def prueba_prof_con_confianza():
    w, h = 4, 2
    profundidad = np.full(w * h, 1.5, "<f2")
    confianza = np.array([0, 1, 2, 2, 2, 2, 1, 0], "u1")
    with tempfile.TemporaryDirectory() as d:
        ruta = Path(d) / "x.prof"
        ruta.write_bytes(struct.pack("<II", w, h) + profundidad.tobytes() + confianza.tobytes())
        mapa = pd.leer_prof(ruta)
    assert mapa.shape == (2, 4)
    assert mapa[0, 0] == 0 and mapa[1, 3] == 0, "las lecturas de confianza baja se descartan"
    assert mapa[0, 1] == 1.5


def prueba_alinear_quita_el_sesgo_y_resiste_atipicos():
    lidar = torch.linspace(1, 3, 192 * 256).reshape(192, 256)
    con_sesgo = (lidar - 0.022) / 1.01
    lidar = lidar.clone()
    lidar[:20] = 0                     # sin dato
    lidar[100:110, :] = 0.3            # objeto delante que la predicción no ve
    alineada, a, b = pd.alinear_a_lidar(con_sesgo, lidar)
    assert abs(a - 1.01) < 0.002 and abs(b - 0.022) < 0.003, (a, b)
    assert (alineada[30:90] - lidar[30:90]).abs().max() < 0.002


def prueba_nombres_de_las_vistas():
    with tempfile.TemporaryDirectory() as d:
        origen = Path(d)
        for sub, archivo in (("fotos_clave", "000"), ("fotogramas", "0001")):
            (origen / sub).mkdir()
            (origen / sub / "camaras.json").write_text(json.dumps({"fotos": [
                {"archivo": f"{archivo}.jpg", "profundidad": {"archivo": f"{archivo}.prof"}}]}), "utf-8")
        reales = pd.prompts_reales(origen)
    assert set(reales) == {"fot_000.jpg", "fot_0001.jpg"}, "mismos nombres que fotogrametria.preparar"


if __name__ == "__main__":
    pruebas = [f for n, f in sorted(globals().items()) if n.startswith("prueba_")]
    for prueba in pruebas:
        prueba()
        print("OK", prueba.__name__)
    print(f"{len(pruebas)} pruebas superadas")
