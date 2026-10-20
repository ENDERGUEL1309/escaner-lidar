"""Imita la función `lanzar` de Modal en local, con modelos falsos, para probar el servidor en modo
GPU_PROVEEDOR=modal sin cuenta de Modal ni GPU. Usa el mismo código de descarga, proceso y subida.

  python pruebas/modal_falso.py [puerto]          (SECRETO_INTERNO igual que en el servidor)

  y en otra terminal:
  GPU_PROVEEDOR=modal MODAL_URL=http://localhost:8790 URL_PUBLICA=http://localhost:8080 SECRETO_INTERNO=… npm start
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import sys
import threading
from http.server import BaseHTTPRequestHandler, ThreadingHTTPServer
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from escaner_ia.trabajo import ejecutar  # noqa: E402
from falsos import ModelosFalsos  # noqa: E402

SECRETO = os.environ.get("SECRETO_INTERNO", "")


class Lanzar(BaseHTTPRequestHandler):
    def do_POST(self):
        cuerpo = self.rfile.read(int(self.headers.get("Content-Length", 0)))
        esperada = hmac.new(SECRETO.encode(), cuerpo, hashlib.sha256).hexdigest()
        if not SECRETO or not hmac.compare_digest(esperada, self.headers.get("X-Firma", "")):
            return self._responder(401, {"detail": "Firma no válida"})
        trabajo = json.loads(cuerpo)
        threading.Thread(target=ejecutar, args=(trabajo, ModelosFalsos(), SECRETO), daemon=True).start()
        self._responder(200, {"llamada": f"fc-falsa-{trabajo['id'][:8]}"})

    def _responder(self, codigo, datos):
        cuerpo = json.dumps(datos).encode()
        self.send_response(codigo)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(cuerpo)))
        self.end_headers()
        self.wfile.write(cuerpo)


if __name__ == "__main__":
    puerto = int(sys.argv[1]) if len(sys.argv) > 1 else 8790
    print(f"Modal falso en http://localhost:{puerto}", flush=True)
    ThreadingHTTPServer(("127.0.0.1", puerto), Lanzar).serve_forever()
