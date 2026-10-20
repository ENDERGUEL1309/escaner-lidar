"""Un trabajo de principio a fin: descargar → procesar → subir → avisar. Es lo que ejecuta la GPU en Modal;
el mismo código serviría con RunPod (solo cambia el envoltorio, ver docs/ia-nube.md §2)."""

from __future__ import annotations

import shutil
import tempfile
import time
import traceback
from pathlib import Path

from . import pipeline
from .comunicacion import Avisador, descargar, subir


def ejecutar(trabajo: dict, modelos, secreto: str | None, raiz: Path | None = None) -> dict:
    ident = str(trabajo["id"])
    opciones = trabajo.get("opciones") or {}
    avisador = Avisador((trabajo.get("progreso") or {}).get("url"), secreto)
    carpeta = Path(tempfile.mkdtemp(prefix=f"trabajo-{ident[:8]}-", dir=raiz))
    entrada, salida = carpeta / "entrada", carpeta / "salida"
    etapas = pipeline.etapas_para(opciones)
    n = len(etapas)
    ultimo = [0.0]

    def avance_descarga(f):
        if time.time() - ultimo[0] > 2:
            ultimo[0] = time.time()
            avisador.etapa(pipeline.ETAPAS["descarga"], f / n)

    try:
        avisador.etapa(pipeline.ETAPAS["descarga"], 0)
        descargar(trabajo.get("entradas") or [], entrada, avance_descarga)
        informe = pipeline.procesar(entrada, salida, opciones, modelos, enviar=avisador.etapa)
        informe["trabajo"] = ident
        avisador.etapa("Subiendo resultados", (n - 0.5) / n)
        subidos = subir(salida, trabajo["salida"]["url"])
        avisador.enviar({"estado": "listo", "progreso": 1, "etapa": None, "error": None}, obligatorio=True)
        return {"ok": True, "archivos": subidos, "informe": informe}
    except Exception as e:
        traceback.print_exc()
        mensaje = f"{type(e).__name__}: {e}"[:500]
        avisador.enviar({"estado": "error", "etapa": None, "error": mensaje})
        return {"ok": False, "error": mensaje}
    finally:
        shutil.rmtree(carpeta, ignore_errors=True)
