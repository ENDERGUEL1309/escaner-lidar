"""App de Modal del trabajador con GPU (sustituye al trabajador simulado de servidor/src/trabajador.js).

  modal deploy app.py                                   publica `lanzar` (la URL que usa el servidor) y la GPU
  modal run app.py --carpeta "C:\\…\\Entorno_…"           prueba con un escaneo local, sin servidor
  modal run app.py --carpeta "…" --sin-fotos            fuerza el modo de prueba con imágenes sintéticas

Instrucciones completas en README.md.
"""

from __future__ import annotations

import hashlib
import hmac
import json
import os
import time
import uuid
from pathlib import Path

import modal

NOMBRE = "escaner-lidar-ia"
SECRETO = "escaner-lidar"          # modal secret create escaner-lidar SECRETO_INTERNO=…
GPU = os.environ.get("ESCANER_GPU", "L4")

app = modal.App(NOMBRE)

# Pesos de Hugging Face en un volumen: se descargan la primera vez y los arranques siguientes son rápidos.
pesos = modal.Volume.from_name("escaner-lidar-pesos", create_if_missing=True)
# Carpetas de `modal run` (pruebas sin servidor): entrada y salida de cada prueba.
pruebas = modal.Volume.from_name("escaner-lidar-pruebas", create_if_missing=True)

imagen_gpu = (
    # Imagen de CUDA con compilador (nvcc): gsplat no publica binarios para PyTorch 2.8 y hay que compilarlo.
    modal.Image.from_registry("nvidia/cuda:12.8.1-devel-ubuntu22.04", add_python="3.12")
    # clang: el Python de Modal se compiló con clang y enlaza las extensiones con clang++.
    .apt_install("git", "libgl1", "libglib2.0-0", "build-essential", "ninja-build", "clang")
    # El Python de esta imagen trae pip antiguo y sin `wheel`: hacen falta para compilar SAM 2 y gsplat.
    .pip_install("pip>=24", "setuptools>=70", "wheel", "ninja")
    .pip_install("torch==2.8.0", "torchvision==0.23.0", index_url="https://download.pytorch.org/whl/cu128")
    .pip_install(
        "numpy==2.2.6", "scipy==1.15.3", "pillow==11.3.0",
        "transformers==4.57.1", "huggingface-hub==0.35.3", "safetensors", "sentencepiece", "protobuf",
        "hydra-core>=1.3.2", "iopath>=0.1.10", "tqdm",
    )
    # SAM 2 solo se publica como código fuente; sin la extensión CUDA opcional (no hace falta para imágenes).
    .env({"SAM2_BUILD_CUDA": "0", "HF_HOME": "/pesos/hf", "TOKENIZERS_PARALLELISM": "false"})
    .pip_install("sam2==1.1.0", extra_options="--no-build-isolation")
    # Gaussian Splatting (etapa F): se compila para las GPU que usamos (A10G = 8.6, L4 = 8.9, L40S = 8.9).
    .env({"TORCH_CUDA_ARCH_LIST": "8.6;8.9", "MAX_JOBS": "8"})
    .pip_install("gsplat==1.5.3", extra_options="--no-build-isolation --no-binary gsplat")
    .add_local_python_source("escaner_ia")
)

imagen_web = modal.Image.debian_slim(python_version="3.12").pip_install("fastapi[standard]==0.118.0")

with imagen_web.imports():
    from fastapi import HTTPException, Request


def _ejecutar(trabajo: dict, secreto: str | None) -> dict:
    from escaner_ia.modelos import Modelos
    from escaner_ia.trabajo import ejecutar

    resultado = ejecutar(trabajo, Modelos(), secreto)
    pesos.commit()        # guarda los pesos descargados en esta ejecución
    return resultado


@app.function(image=imagen_gpu, gpu=GPU, volumes={"/pesos": pesos}, timeout=45 * 60,
              secrets=[modal.Secret.from_name(SECRETO)])
def procesar_trabajo(trabajo: dict) -> dict:
    """Trabajo que llega del servidor: URLs prefirmadas de entrada/salida y URL de progreso."""
    return _ejecutar(trabajo, os.environ["SECRETO_INTERNO"])


@app.function(image=imagen_web, secrets=[modal.Secret.from_name(SECRETO)])
@modal.fastapi_endpoint(method="POST")
async def lanzar(request: Request):
    """Lo llama el servidor (GPU_PROVEEDOR=modal). Comprueba la firma HMAC y lanza la GPU sin esperar."""
    cuerpo = await request.body()
    esperada = hmac.new(os.environ["SECRETO_INTERNO"].encode(), cuerpo, hashlib.sha256).hexdigest()
    if not hmac.compare_digest(esperada, request.headers.get("x-firma", "")):
        raise HTTPException(status_code=401, detail="Firma no válida")
    trabajo = json.loads(cuerpo)
    if abs(time.time() * 1000 - float(trabajo.get("emitido", 0))) > 10 * 60 * 1000:
        raise HTTPException(status_code=400, detail="Petición caducada (revisa la hora del servidor)")
    llamada = await procesar_trabajo.spawn.aio(trabajo)
    return {"llamada": llamada.object_id}


# ------------------------------------------------------------------------------ pruebas con `modal run`

@app.function(image=imagen_gpu, gpu=GPU, volumes={"/pesos": pesos, "/pruebas": pruebas}, timeout=45 * 60)
def procesar_prueba(carpeta: str, opciones: dict) -> dict:
    """Igual que `procesar_trabajo`, pero leyendo y escribiendo en el volumen de pruebas."""
    from escaner_ia import pipeline
    from escaner_ia.modelos import Modelos

    pruebas.reload()
    raiz = Path("/pruebas") / carpeta
    informe = pipeline.procesar(raiz / "entrada", raiz / "salida", opciones, Modelos(),
                                enviar=lambda etapa, p: print(f"[{p:5.0%}] {etapa}", flush=True))
    pesos.commit()
    pruebas.commit()
    return informe


SUBIR = ("malla_color.ply", "escena.json", "textura.jpg", "edicion.json")


@app.local_entrypoint()
def prueba(carpeta: str, salida: str = "", sin_fotos: bool = False, vocabulario: str = "", splats: bool = False,
           sin_segmentacion: bool = False):
    """Procesa la carpeta de un escaneo del iPhone y deja el resultado en <carpeta>/ia (o en --salida)."""
    origen = Path(carpeta).expanduser().resolve()
    if not origen.is_dir():
        raise SystemExit(f"No existe la carpeta {origen}")
    destino = Path(salida).expanduser().resolve() if salida else origen / "ia"
    remoto = f"{origen.name[:60]}-{uuid.uuid4().hex[:8]}"

    archivos = [origen / n for n in SUBIR if (origen / n).is_file()]
    archivos += sorted((origen / "objetos").glob("*.ply")) + sorted((origen / "fotos_clave").glob("*"))
    archivos += sorted((origen / "fotogramas").glob("*"))
    if not any(a.name in ("malla_color.ply", "escena.json") for a in archivos):
        raise SystemExit("La carpeta no parece un escaneo: falta malla_color.ply o escena.json")
    total = sum(a.stat().st_size for a in archivos) / 1e6
    print(f"Subiendo {len(archivos)} archivos ({total:.0f} MB) al volumen de pruebas…")
    with pruebas.batch_upload(force=True) as lote:
        for a in archivos:
            lote.put_file(str(a), f"/{remoto}/entrada/{a.relative_to(origen).as_posix()}")

    opciones = {"segmentacion": not sin_segmentacion, "splats": splats,
                "vistas": "sinteticas" if sin_fotos else "auto",
                "vocabulario": [p.strip() for p in vocabulario.split(",") if p.strip()]}
    informe = procesar_prueba.remote(remoto, opciones)

    print(f"Descargando resultados en {destino}…")
    for entrada in pruebas.listdir(f"/{remoto}/salida", recursive=True):
        if entrada.type != modal.volume.FileEntryType.FILE:
            continue
        relativa = Path(entrada.path.lstrip("/")).relative_to(f"{remoto}/salida")
        local = destino / relativa
        local.parent.mkdir(parents=True, exist_ok=True)
        with open(local, "wb") as f:
            for trozo in pruebas.read_file(entrada.path):
                f.write(trozo)
    pruebas.remove_file(f"/{remoto}", recursive=True)
    print(json.dumps({k: informe.get(k) for k in ("vistas", "mascaras2d", "instancias", "objetos", "splats", "tiempos", "avisos")},
                     ensure_ascii=False, indent=2))
    print(f"Listo: {destino / 'escena.json'}")
