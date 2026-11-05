r"""Trabajador del PC: tu GPU procesa los escaneos que mandas desde el iPhone.

    trabajador_pc.bat            (lo deja arrancado instalar_trabajador_pc.bat al iniciar Windows)

Pregunta al servidor cada pocos segundos (POST /api/pc/tomar con el API_TOKEN) si hay un escaneo marcado como
«Malla de alta calidad». Si lo hay: lo descarga, ejecuta procesar_escaneo (fotogrametría, 2DGS, malla y planos),
sube la malla para el iPhone y avisa de que está listo. Usa las mismas URLs firmadas y avisos de progreso que la
GPU de Modal (escaner_ia/comunicacion.py).

Configuración (por orden): variables de entorno ESCANER_SERVIDOR, API_TOKEN, SECRETO_INTERNO, o los archivos
servidor/.api-token y servidor/.secreto-interno de este repositorio.

Al arrancar abre el panel local (panel_pc.py) en http://localhost:8770 con el progreso, la GPU y el registro.
Sin navegador: variable PANEL_PC_NAVEGADOR=0.
"""

from __future__ import annotations

import json
import os
import re
import shutil
import sys
import threading
import time
import traceback
import urllib.error
import urllib.request
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))
from escaner_ia.comunicacion import Avisador, descargar, subir  # noqa: E402
from panel_pc import Panel, arrancar as arrancar_panel  # noqa: E402

REPO = Path(__file__).resolve().parent.parent
SERVIDOR = os.environ.get("ESCANER_SERVIDOR", "https://servidor-production-784c.up.railway.app").rstrip("/")
CARPETA = Path.home() / "EscaneosPC"
ESPERA_S = 15
CARPETA.mkdir(exist_ok=True)
PANEL = Panel(CARPETA)


def _secreto(variable: str, archivo: str) -> str:
    valor = os.environ.get(variable) or (REPO / "servidor" / archivo).read_text().strip()
    if not valor:
        raise SystemExit(f"Falta {variable} (o servidor/{archivo})")
    return valor


TOKEN = _secreto("API_TOKEN", ".api-token")
SECRETO = _secreto("SECRETO_INTERNO", ".secreto-interno")


def tomar() -> dict | None:
    peticion = urllib.request.Request(f"{SERVIDOR}/api/pc/tomar", data=b"", method="POST",
                                      headers={"Authorization": f"Bearer {TOKEN}"})
    with urllib.request.urlopen(peticion, timeout=60) as r:
        return json.loads(r.read()) if r.status == 200 else None


class Progreso:
    """Traduce las líneas del proceso a etapa y porcentaje para la app (como mucho un aviso cada 5 s)."""

    ETAPAS = [  # (patrón en la línea, progreso, texto para la app)
        (r"Fusión LiDAR", 0.04, "Fundiendo el LiDAR de la grabación"),
        (r"Preparando im", 0.05, "Preparando las fotos"),
        (r"Fotogrametría reutilizada", 0.13, "Reutilizando la fotogrametría de un envío anterior"),
        (r"feature_extractor", 0.06, "Buscando puntos en las fotos"),
        (r"exhaustive_matcher", 0.08, "Emparejando las fotos"),
        (r"point_triangulator|bundle_adjuster", 0.11, "Corrigiendo las posiciones de la cámara"),
        (r"image_undistorter", 0.13, "Preparando la profundidad densa"),
        (r"patch_match_stereo", 0.14, "Calculando la profundidad píxel a píxel (lo más largo)"),
        (r"stereo_fusion", 0.30, "Fusionando la nube densa"),
        (r"Alineación con el LiDAR", 0.32, "Alineando con el LiDAR"),
        (r"Entrenando", 0.33, "Entrenando la malla 3D (Gaussian Splatting 2D)"),
        (r"Rejilla TSDF", 0.60, "Extrayendo la malla"),
        (r"Giro de la habitación:", 0.62, "Ajustando paredes, suelo y techo"),
        (r"Versión ligera", 0.64, "Preparando la versión para el iPhone"),
        (r"Modelado verificado", 0.66, "Modelando los muebles y comprobándolos con las fotos"),
        (r"Render fotorrealista", 0.69, "Creando el render fotorrealista"),
        (r"8/8 Remodelado: paredes", 0.94, "Remodelando: paredes, suelo y techo exactos"),
        (r"8/8 Remodelado: muebles", 0.95, "Remodelando: separando los muebles"),
        (r"8/8 Remodelado: generando", 0.955, "Remodelando: generando los muebles con Hunyuan3D-2"),
        (r"8/8 Remodelado: encajando", 0.97, "Remodelando: comprobando cada mueble con el LiDAR"),
        (r"8/8 Remodelado: montando", 0.975, "Remodelando: montando la escena en Blender"),
    ]
    # Tramos que avanzan con el número de paso del registro: (etapa, patrón, desde, hasta, total de pasos).
    ENTRENO = (0.33, 0.59)
    RENDER = (0.70, 0.93, 30_000)

    # La profundidad densa de COLMAP (solo con --densa) no escribe nada en ~45 min: su avance se estima por el tiempo.
    DENSA_S, DENSA_DESDE, DENSA_HASTA = 45 * 60, 0.14, 0.29

    def __init__(self, avisador: Avisador, pasos: int):
        self.avisador, self.pasos = avisador, pasos
        self.etapa, self.valor, self.ultimo, self.inicio_etapa = "Procesando en tu PC", 0.0, 0.0, time.time()
        self.cerrojo = threading.Lock()
        self.activo = True
        threading.Thread(target=self._latido, daemon=True).start()

    def _latido(self):
        while self.activo:
            time.sleep(30)
            with self.cerrojo:
                if self.etapa.startswith("Calculando la profundidad"):
                    t = min(1.0, (time.time() - self.inicio_etapa) / self.DENSA_S)
                    self.valor = max(self.valor, self.DENSA_DESDE + (self.DENSA_HASTA - self.DENSA_DESDE) * t)
                if self.activo and time.time() - self.ultimo > 25:
                    self._enviar()

    def _enviar(self):
        self.ultimo = time.time()
        self.avisador.etapa(self.etapa, self.valor)
        PANEL.valor(self.valor)

    def terminar(self):
        self.activo = False

    def __call__(self, linea: str):
        print(linea, flush=True)
        PANEL.linea(linea)
        with self.cerrojo:
            anterior = self.etapa
            for patron, valor, texto in self.ETAPAS:
                if re.search(patron, linea):
                    self.etapa, self.valor = texto, max(self.valor, valor)
            m = re.search(r"paso\s+(\d+)", linea)
            if m and self.etapa.startswith("Entrenando"):
                desde, hasta = self.ENTRENO
                self.valor = max(self.valor, desde + (hasta - desde) * int(m.group(1)) / self.pasos)
            m = re.search(r"render\s+(\d+)\s+L1", linea)
            if m and self.etapa.startswith("Creando el render"):
                desde, hasta, total = self.RENDER
                self.valor = max(self.valor, desde + (hasta - desde) * int(m.group(1)) / total)
            if self.etapa != anterior:
                PANEL.etapa(self.etapa, self.valor)
            if self.etapa != anterior:
                self.inicio_etapa = time.time()
            # Un cambio de etapa se avisa enseguida; el resto, como mucho cada 5 s.
            if self.etapa != anterior or time.time() - self.ultimo > 5:
                self._enviar()


def atender(trabajo: dict):
    from procesar_escaneo import procesar

    avisador = Avisador(trabajo["progreso"]["url"], SECRETO)
    nombre = re.sub(r"[^\w.-]+", "_", trabajo.get("nombre") or "Escaneo")[:60]
    carpeta = CARPETA / f"{nombre}_{trabajo['id'][:8]}"
    print(f"\n=== Trabajo {trabajo['id']} ({nombre}) → {carpeta}", flush=True)
    PANEL.nuevo(nombre, carpeta)
    try:
        avisador.etapa("Descargando en tu PC", 0.0)
        PANEL.etapa("Descargando en tu PC", 0.0)
        # La descarga es el 0–3 % de la barra; se avisa como mucho cada 5 s para no saturar el servidor.
        ultimo_aviso = [0.0]

        def avance_descarga(fraccion: float):
            if time.time() - ultimo_aviso[0] >= 5 or fraccion >= 1:
                ultimo_aviso[0] = time.time()
                avisador.etapa(f"Descargando en tu PC ({round(fraccion * 100)} %)", 0.03 * fraccion)
            PANEL.etapa(f"Descargando en tu PC ({round(fraccion * 100)} %)", 0.03 * fraccion)

        descargar(trabajo["entradas"], carpeta, avance_descarga)
        pasos = 20_000
        progreso = Progreso(avisador, pasos)
        try:
            destino = procesar(carpeta, pasos=pasos, registro=progreso)
        finally:
            progreso.terminar()
        subida = carpeta / "para_el_iphone"
        subida.mkdir(exist_ok=True)
        shutil.copy2(destino / "malla_movil.ply", subida / "malla_alta.ply")
        if (destino / "escena_modelada.ply").exists():
            shutil.copy2(destino / "escena_modelada.ply", subida / "escena_modelada.ply")
        # Habitación remodelada: USDZ para verla en el iPhone (3D y realidad aumentada) y su vista previa.
        for nombre in ("habitacion_remodelada.usdz", "vista_previa.png"):
            if (carpeta / "remodelado" / nombre).exists():
                shutil.copy2(carpeta / "remodelado" / nombre, subida / nombre)
        render = carpeta / "splats_render" / "escena.splat"
        if render.exists():
            (subida / "splats").mkdir(exist_ok=True)
            shutil.copy2(render, subida / "splats" / "escena.splat")
        avisador.etapa("Subiendo la malla", 0.99)
        PANEL.etapa("Subiendo los resultados al iPhone", 0.98)
        subir(subida, trabajo["salida"]["url"])
        avisador.enviar({"estado": "listo", "progreso": 1, "etapa": "Listo", "error": None}, obligatorio=True)
        PANEL.fin(True)
        print(f"=== Listo: {trabajo['id']}", flush=True)
    except Exception as error:  # noqa: BLE001 — cualquier fallo se comunica a la app
        traceback.print_exc()
        mensaje = str(error).strip().splitlines()[-1] if str(error).strip() else type(error).__name__
        PANEL.fin(False, mensaje[:300])
        avisador.enviar({"estado": "error", "etapa": None, "error": f"Falló en el PC: {mensaje[:300]}"},
                        obligatorio=True)


def main():
    print(f"Trabajador del PC conectado a {SERVIDOR}. Esperando escaneos (Ctrl+C para salir)…", flush=True)
    arrancar_panel(PANEL, abrir_navegador=os.environ.get("PANEL_PC_NAVEGADOR", "1") != "0")
    avisado = False
    while True:
        try:
            trabajo = tomar()
            avisado = False
            PANEL.conexion(SERVIDOR, True)
        except (urllib.error.URLError, TimeoutError, ConnectionError) as error:
            PANEL.conexion(SERVIDOR, False)
            if not avisado:
                print(f"Sin conexión con el servidor ({error}); reintentando…", flush=True)
                avisado = True
            time.sleep(60)
            continue
        if trabajo:
            atender(trabajo)
        else:
            time.sleep(ESPERA_S)


if __name__ == "__main__":
    import faulthandler
    faulthandler.enable()
    main()
