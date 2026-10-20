"""Descarga del paquete, subida de resultados y avisos de progreso firmados (solo biblioteca estándar).

El servidor (o R2/S3 más adelante) da URLs prefirmadas:
  trabajo["entradas"] = [{"ruta": "malla_color.ply", "url": "https://…"}, …]   GET de cada archivo
  trabajo["salida"]   = {"url": "https://…/salida/{ruta}?…"}                   PUT; {ruta} se sustituye
  trabajo["progreso"] = {"url": "https://…/api/interno/trabajos/<id>/progreso"} POST con X-Firma
"""

from __future__ import annotations

import hashlib
import hmac
import json
import shutil
import time
import urllib.error
import urllib.parse
import urllib.request
from pathlib import Path

TIEMPO = 120


def firma(secreto: str, cuerpo: bytes) -> str:
    return hmac.new(secreto.encode(), cuerpo, hashlib.sha256).hexdigest()


def _reintentar(fn, intentos=4):
    for i in range(intentos):
        try:
            return fn()
        except urllib.error.HTTPError as e:
            if e.code < 500 and e.code != 429:
                detalle = e.read()[:300].decode("utf-8", "replace")
                raise RuntimeError(f"HTTP {e.code}: {detalle}") from None
            if i == intentos - 1:
                raise
        except (urllib.error.URLError, TimeoutError, ConnectionError):
            if i == intentos - 1:
                raise
        time.sleep(2 ** i)


def ruta_segura(ruta: str) -> str:
    partes = ruta.replace("\\", "/").split("/")
    if not ruta or ruta.startswith("/") or any(p in ("", ".", "..") for p in partes):
        raise ValueError(f"Ruta no válida: {ruta!r}")
    return "/".join(partes)


def descargar(entradas: list[dict], carpeta: Path, avance=lambda f: None) -> None:
    for i, e in enumerate(entradas):
        destino = carpeta / ruta_segura(e["ruta"])
        destino.parent.mkdir(parents=True, exist_ok=True)

        def bajar():
            with urllib.request.urlopen(e["url"], timeout=TIEMPO) as r, open(destino, "wb") as f:
                shutil.copyfileobj(r, f, 1 << 20)

        _reintentar(bajar)
        avance((i + 1) / len(entradas))


def subir(carpeta: Path, plantilla: str, avance=lambda f: None) -> list[str]:
    archivos = sorted(p for p in carpeta.rglob("*") if p.is_file())
    rutas = []
    for i, archivo in enumerate(archivos):
        ruta = archivo.relative_to(carpeta).as_posix()
        url = plantilla.replace("{ruta}", "/".join(urllib.parse.quote(p) for p in ruta.split("/")))

        def poner():
            with open(archivo, "rb") as f:
                peticion = urllib.request.Request(url, data=f, method="PUT", headers={
                    "Content-Length": str(archivo.stat().st_size), "Content-Type": "application/octet-stream"})
                with urllib.request.urlopen(peticion, timeout=TIEMPO) as r:
                    r.read()

        _reintentar(poner)
        rutas.append(ruta)
        avance((i + 1) / len(archivos))
    return rutas


class Avisador:
    """POST /api/interno/trabajos/<id>/progreso con HMAC-SHA256 del cuerpo en X-Firma.
    Un fallo al avisar no detiene el trabajo (salvo el aviso final, que se reintenta)."""

    def __init__(self, url: str | None, secreto: str | None):
        self.url, self.secreto = url, secreto

    def enviar(self, datos: dict, obligatorio=False):
        print(f"[progreso] {json.dumps(datos, ensure_ascii=False)}", flush=True)
        if not self.url:
            return
        cuerpo = json.dumps(datos, ensure_ascii=False).encode()
        peticion = urllib.request.Request(self.url, data=cuerpo, method="POST", headers={
            "Content-Type": "application/json", "X-Firma": firma(self.secreto or "", cuerpo)})
        try:
            _reintentar(lambda: urllib.request.urlopen(peticion, timeout=30).read(), 4 if obligatorio else 2)
        except Exception as e:
            if obligatorio:
                raise
            print(f"[progreso] no se pudo avisar: {e}", flush=True)

    def etapa(self, texto: str, progreso: float):
        self.enviar({"estado": "procesando", "etapa": texto, "progreso": round(progreso, 4)})
