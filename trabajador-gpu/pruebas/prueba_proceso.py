"""Prueba del proceso completo con modelos falsos (sin GPU):  python pruebas/prueba_proceso.py

Comprueba que la caja roja y la azul, que el iPhone dejó juntas en «Objeto 1», salen como dos objetos
(«Silla 1» y «Mesa 1»), que la estructura se conserva y que los PLY tienen el formato de la app.
"""

from __future__ import annotations

import json
import sys
import tempfile
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escaner_ia import camaras, pipeline, ply  # noqa: E402
from falsos import ModelosFalsos, crear_escaneo  # noqa: E402


def prueba_ply(tmp: Path):
    m = ply.Malla(posiciones=np.random.rand(10, 3).astype(np.float32), normales=np.zeros((10, 3), np.float32),
                  colores=np.full((10, 3), 7, np.uint8), caras=np.array([[0, 1, 2], [2, 3, 4]]),
                  st=np.random.rand(10, 2).astype(np.float32), textura="../textura.jpg")
    ply.escribir(m, tmp / "a.ply")
    r = ply.leer(tmp / "a.ply")
    assert np.allclose(r.posiciones, m.posiciones) and np.allclose(r.st, m.st)
    assert (r.caras == m.caras).all() and (r.inferido == 0).all() and r.textura == "../textura.jpg"
    cab = (tmp / "a.ply").read_bytes().split(b"end_header")[0].decode()
    props = [l.split()[-1] for l in cab.splitlines() if l.startswith("property float") or l.startswith("property uchar")]
    assert props == ["x", "y", "z", "nx", "ny", "nz", "s", "t", "red", "green", "blue", "inferido"], props


def prueba_camaras():
    # Cámara ARKit en (0, 1, 2) mirando hacia −Z (la identidad de ARKit mira a −Z): el origen queda delante.
    c2w = np.eye(4)
    c2w[:3, 3] = [0, 1, 2]
    m = camaras.arkit_a_opencv(camaras.matriz_columnas(c2w.T.ravel()))
    v = camaras.Vista("x", np.zeros((480, 640, 3), np.uint8), 500, 500, 319.5, 239.5, m)
    u, vv, z = v.proyectar(np.array([[0, 1, 0], [0.1, 1.1, 0]], np.float32))
    assert np.allclose(z, 2) and np.allclose(u[0], 319.5) and np.allclose(vv[0], 239.5)
    assert u[1] > u[0] and vv[1] < vv[0]          # x a la derecha, y arriba en el mundo → arriba en la imagen
    w2c = camaras.mirar_a([0, 1, 2], [0, 1, 0])
    assert np.allclose(w2c, m, atol=1e-9), (w2c, m)


def prueba_proceso(tmp: Path):
    entrada = crear_escaneo(tmp / "entrada")
    salida = tmp / "salida"
    avisos = []
    t0 = time.time()
    informe = pipeline.procesar(entrada, salida, {"segmentacion": True, "vistas": "auto"}, ModelosFalsos(),
                                enviar=lambda e, p: avisos.append((e, p)))
    print(f"  proceso en {time.time() - t0:.1f} s; vistas {informe['vistas']}, máscaras {informe['mascaras2d']}, "
          f"instancias {informe['instancias']}")
    escena = json.loads((salida / "escena.json").read_text("utf-8"))
    nombres = [o["nombre"] for o in escena["objetos"]]
    print("  objetos:", nombres)
    assert escena["version"] == 2
    assert nombres[:2] == ["Suelo", "Paredes"], nombres
    assert "Silla 1" in nombres and "Mesa 1" in nombres, nombres
    por_nombre = {o["nombre"]: o for o in escena["objetos"]}
    silla, mesa = por_nombre["Silla 1"], por_nombre["Mesa 1"]
    assert silla["categoria"] == "asiento" and mesa["categoria"] == "mesa"
    assert silla["fuente"] == "nube" and silla["confianzaIA"] > 0.5, silla
    # Caja roja: 0,5 × 0,5 × 0,5 centrada en x = −0,25. Caja azul: 0,6 × 0,7 × 0,6 en x = 0,3.
    assert abs(silla["centro"][0] + 0.25) < 0.06 and abs(mesa["centro"][0] - 0.3) < 0.06, (silla, mesa)
    assert abs(mesa["dimensiones"][1] - 0.7) < 0.06, mesa
    for o in escena["objetos"]:
        m = ply.leer(salida / o["archivo"])
        assert m.st is not None and m.textura == "../textura.jpg" and m.n_vertices == o["vertices"], o
    rojo = ply.leer(salida / silla["archivo"]).colores.mean(0)
    assert rojo[0] > 150 and rojo[2] < 90, rojo
    assert (salida / "textura.jpg").exists() and (salida / "informe.json").exists()
    progresos = [p for _, p in avisos]
    assert progresos == sorted(progresos) and progresos[-1] < 1, progresos
    return escena


if __name__ == "__main__":
    with tempfile.TemporaryDirectory() as d:
        tmp = Path(d)
        prueba_ply(tmp)
        print("ok  PLY")
        prueba_camaras()
        print("ok  cámaras")
        prueba_proceso(tmp)
        print("ok  proceso completo")
