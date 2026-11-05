r"""Panel local del trabajador del PC: una página en el navegador con lo que está pasando.

trabajador_pc.py lo arranca solo y abre http://localhost:8770. Muestra el escaneo en curso (barra, etapa, tiempo
y estimación de lo que falta), cada etapa con su duración, la GPU (uso, memoria y temperatura), el registro en vivo,
la comparación foto/render al terminar y los últimos trabajos. Desde la página se puede abrir la carpeta del
escaneo o cargar la malla en Blender.

Solo escucha en 127.0.0.1: no es accesible desde otros equipos. Las acciones exigen la cabecera X-Panel, que una
web ajena no puede enviar sin permiso del navegador.
"""

from __future__ import annotations

import collections
import json
import os
import shutil
import subprocess
import threading
import time
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

PUERTO = int(os.environ.get("PANEL_PC_PUERTO", "8770"))


class Panel:
    def __init__(self, carpeta_historial: Path):
        self.cerrojo = threading.Lock()
        self.ruta_historial = carpeta_historial / "historial_pc.json"
        try:
            self.historial = json.loads(self.ruta_historial.read_text("utf-8"))
        except (OSError, ValueError):
            self.historial = []
        self.servidor_ok = None
        self.servidor_url = ""
        self.trabajo = None          # dict del trabajo en curso
        self.ultimo = None           # último trabajo terminado (para su imagen y sus botones)
        self.registro = collections.deque(maxlen=400)
        self._gpu, self._gpu_t = None, 0.0

    # ------------------------------------------------------------------ eventos del trabajador

    def conexion(self, url: str, ok: bool):
        with self.cerrojo:
            self.servidor_url, self.servidor_ok = url, ok

    def nuevo(self, nombre: str, carpeta: Path):
        with self.cerrojo:
            self.registro.clear()
            self.trabajo = {"nombre": nombre, "carpeta": str(carpeta), "inicio": time.time(), "valor": 0.0,
                            "etapa": "Empezando", "etapas": [], "estado": "procesando", "error": None}

    def etapa(self, texto: str, valor: float):
        with self.cerrojo:
            t = self.trabajo
            if not t:
                return
            t["valor"] = max(t["valor"], valor)
            base = texto.split(" (")[0]          # «Descargando en tu PC (40 %)» es la misma etapa que al 10 %
            if not t["etapas"] or t["etapas"][-1]["base"] != base:
                ahora = time.time()
                if t["etapas"]:
                    t["etapas"][-1]["fin"] = ahora
                t["etapas"].append({"base": base, "texto": texto, "inicio": ahora, "fin": None})
            else:
                t["etapas"][-1]["texto"] = texto
            t["etapa"] = texto

    def valor(self, valor: float):
        with self.cerrojo:
            if self.trabajo:
                self.trabajo["valor"] = max(self.trabajo["valor"], valor)

    def linea(self, texto: str):
        with self.cerrojo:
            self.registro.append(time.strftime("%H:%M:%S ") + texto)

    def fin(self, ok: bool, error: str | None = None):
        with self.cerrojo:
            t = self.trabajo
            if not t:
                return
            ahora = time.time()
            if t["etapas"] and not t["etapas"][-1]["fin"]:
                t["etapas"][-1]["fin"] = ahora
            t.update(estado="listo" if ok else "error", error=error, fin=ahora, valor=1.0 if ok else t["valor"])
            calidad = Path(t["carpeta"]) / "splats_render" / "calidad.json"
            try:
                t["calidad"] = json.loads(calidad.read_text())
            except (OSError, ValueError):
                t["calidad"] = None
            self.historial.insert(0, {"nombre": t["nombre"], "carpeta": t["carpeta"], "inicio": t["inicio"],
                                      "minutos": round((ahora - t["inicio"]) / 60, 1), "estado": t["estado"],
                                      "error": error, "psnr": (t["calidad"] or {}).get("psnr_reservadas")})
            self.historial = self.historial[:20]
            try:
                self.ruta_historial.write_text(json.dumps(self.historial, ensure_ascii=False, indent=1), "utf-8")
            except OSError:
                pass
            self.ultimo, self.trabajo = t, None

    # ------------------------------------------------------------------ datos para la página

    def gpu(self) -> dict | None:
        if time.time() - self._gpu_t < 2:
            return self._gpu
        self._gpu_t = time.time()
        try:
            salida = subprocess.run(
                ["nvidia-smi", "--query-gpu=name,utilization.gpu,memory.used,memory.total,temperature.gpu",
                 "--format=csv,noheader,nounits"], capture_output=True, text=True, timeout=5,
                creationflags=getattr(subprocess, "CREATE_NO_WINDOW", 0)).stdout.strip().splitlines()[0]
            nombre, uso, usada, total, temp = [x.strip() for x in salida.split(",")]
            self._gpu = {"nombre": nombre, "uso": int(uso), "usada": int(usada), "total": int(total), "temp": int(temp)}
        except Exception:  # noqa: BLE001 — sin nvidia-smi el panel funciona igual
            self._gpu = None
        return self._gpu

    def estado(self) -> dict:
        with self.cerrojo:
            t = self.trabajo or self.ultimo
            datos = {"ahora": time.time(), "servidor": {"url": self.servidor_url, "ok": self.servidor_ok},
                     "trabajo": dict(t) if t else None, "en_curso": self.trabajo is not None,
                     "registro": list(self.registro)[-200:], "historial": self.historial}
        if t:
            datos["trabajo"]["etapas"] = [dict(e) for e in t["etapas"]]
            carpeta = Path(t["carpeta"])
            datos["trabajo"]["imagen"] = (carpeta / "splats_render" / "comparacion.jpg").exists()
            datos["trabajo"]["malla"] = self.malla(carpeta) is not None
        datos["gpu"] = self.gpu()
        return datos

    @staticmethod
    def malla(carpeta: Path) -> Path | None:
        for nombre in ("splat2d/malla_2dgs_plana.ply", "splat2d/malla_movil.ply", "malla_color.ply"):
            if (carpeta / nombre).exists():
                return carpeta / nombre
        return None

    # ------------------------------------------------------------------ acciones

    def carpeta_actual(self) -> Path | None:
        with self.cerrojo:
            t = self.trabajo or self.ultimo
            return Path(t["carpeta"]) if t else None

    def abrir_carpeta(self) -> str:
        carpeta = self.carpeta_actual()
        if not carpeta or not carpeta.exists():
            return "No hay carpeta todavía"
        os.startfile(carpeta)  # noqa: S606 — carpeta del propio trabajador
        return "Carpeta abierta"

    @staticmethod
    def blender_exe() -> Path | None:
        if os.environ.get("BLENDER"):
            return Path(os.environ["BLENDER"])
        encontrado = shutil.which("blender")
        if encontrado:
            return Path(encontrado)
        candidatos = sorted(Path(os.environ.get("ProgramFiles", r"C:\Program Files")).glob(
            "Blender Foundation/Blender*/blender.exe"), reverse=True)
        return candidatos[0] if candidatos else None

    def abrir_blender(self) -> str:
        carpeta = self.carpeta_actual()
        malla = self.malla(carpeta) if carpeta else None
        exe = self.blender_exe()
        if not malla:
            return "Todavía no hay malla"
        if not exe:
            return "No encuentro Blender (define la variable BLENDER con la ruta de blender.exe)"
        # Importa la malla con sus colores por vértice y deja la vista en modo «color del atributo».
        codigo = (
            "import bpy\n"
            "for o in list(bpy.data.objects): bpy.data.objects.remove(o)\n"
            f"bpy.ops.wm.ply_import(filepath=r'{malla}')\n"
            "for area in (bpy.context.screen.areas if bpy.context.screen else []):\n"
            "    if area.type == 'VIEW_3D':\n"
            "        s = area.spaces.active.shading\n"
            "        s.type = 'SOLID'\n"
            "        s.color_type = 'VERTEX'\n"
            "        with bpy.context.temp_override(area=area, region=area.regions[-1]):\n"
            "            bpy.ops.view3d.view_all()\n"
        )
        subprocess.Popen([str(exe), "--python-expr", codigo])
        return f"Abriendo {malla.name} en Blender…"

    def imagen(self) -> bytes | None:
        carpeta = self.carpeta_actual()
        ruta = carpeta / "splats_render" / "comparacion.jpg" if carpeta else None
        return ruta.read_bytes() if ruta and ruta.exists() else None


def arrancar(panel: Panel, abrir_navegador: bool = True) -> str | None:
    pagina = (Path(__file__).parent / "panel_pc.html").read_bytes()

    class Manejador(BaseHTTPRequestHandler):
        def log_message(self, *_):
            pass

        def _enviar(self, cuerpo: bytes, tipo: str, codigo: int = 200):
            self.send_response(codigo)
            self.send_header("Content-Type", tipo)
            self.send_header("Content-Length", str(len(cuerpo)))
            self.send_header("Cache-Control", "no-store")
            self.end_headers()
            self.wfile.write(cuerpo)

        def do_GET(self):
            ruta = self.path.split("?")[0]
            if ruta == "/":
                self._enviar(pagina, "text/html; charset=utf-8")
            elif ruta == "/estado":
                self._enviar(json.dumps(panel.estado(), ensure_ascii=False).encode(), "application/json")
            elif ruta == "/comparacion.jpg":
                datos = panel.imagen()
                self._enviar(datos or b"", "image/jpeg", 200 if datos else 404)
            else:
                self._enviar(b"", "text/plain", 404)

        def do_POST(self):
            if self.headers.get("X-Panel") != "1":
                return self._enviar(b"", "text/plain", 403)
            acciones = {"/accion/carpeta": panel.abrir_carpeta, "/accion/blender": panel.abrir_blender}
            accion = acciones.get(self.path)
            if not accion:
                return self._enviar(b"", "text/plain", 404)
            try:
                mensaje = accion()
            except Exception as error:  # noqa: BLE001
                mensaje = f"No se pudo: {error}"
            self._enviar(json.dumps({"mensaje": mensaje}, ensure_ascii=False).encode(), "application/json")

    try:
        servidor = ThreadingHTTPServer(("127.0.0.1", PUERTO), Manejador)
    except OSError as error:
        print(f"Panel local no disponible en el puerto {PUERTO}: {error}", flush=True)
        return None
    threading.Thread(target=servidor.serve_forever, daemon=True).start()
    url = f"http://localhost:{PUERTO}"
    print(f"Panel del trabajador: {url}", flush=True)
    if abrir_navegador:
        import webbrowser
        webbrowser.open(url)
    return url
