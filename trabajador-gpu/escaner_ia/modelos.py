"""Modelos de IA reales (solo se importan dentro del contenedor con GPU).

Interfaz que usa el resto del código (las pruebas la sustituyen por modelos falsos):
  detector.detectar(imagen, frases)        → [(caja xyxy, puntuación, índice de frase)]
  segmentador.segmentar(imagen, cajas)     → máscaras bool (n, H, W)
  nombrador.vectores_imagen([PIL])         → (n, D) normalizados
  nombrador.vectores_texto([str])          → (m, D) normalizados
  nombrador.escala                         → temperatura de los logits (logit_scale.exp())
"""

from __future__ import annotations

import numpy as np

DINO = "IDEA-Research/grounding-dino-base"
SAM2 = "facebook/sam2.1-hiera-large"
SIGLIP = "google/siglip2-so400m-patch14-384"

FRASES_POR_LLAMADA = 16   # Grounding DINO admite 256 tokens de texto


class Detector:
    def __init__(self, umbral_caja=0.30, umbral_texto=0.25):
        import torch
        from transformers import AutoModelForZeroShotObjectDetection, AutoProcessor

        self.torch = torch
        self.procesador = AutoProcessor.from_pretrained(DINO)
        self.modelo = AutoModelForZeroShotObjectDetection.from_pretrained(DINO).to("cuda").eval()
        self.umbral_caja, self.umbral_texto = umbral_caja, umbral_texto

    def detectar(self, imagen: np.ndarray, frases: list[str]):
        from PIL import Image

        from .vocabulario import Entrada, buscar

        pil = Image.fromarray(imagen)
        entradas = [Entrada(f, f, "objeto") for f in frases]
        resultado = []
        for inicio in range(0, len(frases), FRASES_POR_LLAMADA):
            grupo = frases[inicio:inicio + FRASES_POR_LLAMADA]
            texto = " . ".join(grupo) + " ."
            entradas_modelo = self.procesador(images=pil, text=texto, return_tensors="pt").to("cuda")
            with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.float16):
                salida = self.modelo(**entradas_modelo)
            r = self.procesador.post_process_grounded_object_detection(
                salida, entradas_modelo.input_ids, threshold=self.umbral_caja, text_threshold=self.umbral_texto,
                target_sizes=[(pil.height, pil.width)])[0]
            etiquetas = r.get("text_labels", r.get("labels"))
            for caja, punt, etiqueta in zip(r["boxes"].float().cpu().numpy(), r["scores"].float().cpu().numpy(), etiquetas):
                i = buscar(entradas[inicio:inicio + len(grupo)], str(etiqueta))
                if i is not None:
                    resultado.append((caja, float(punt), inicio + i))
        return resultado


class Segmentador:
    def __init__(self):
        import torch
        from sam2.sam2_image_predictor import SAM2ImagePredictor

        self.torch = torch
        self.predictor = SAM2ImagePredictor.from_pretrained(SAM2, device="cuda")

    def segmentar(self, imagen: np.ndarray, cajas: np.ndarray) -> np.ndarray:
        if len(cajas) == 0:
            return np.zeros((0,) + imagen.shape[:2], bool)
        with self.torch.inference_mode(), self.torch.autocast("cuda", dtype=self.torch.bfloat16):
            self.predictor.set_image(imagen)
            mascaras = []
            for inicio in range(0, len(cajas), 32):
                m, _, _ = self.predictor.predict(box=np.asarray(cajas[inicio:inicio + 32], np.float32),
                                                 multimask_output=False)
                m = np.asarray(m)
                if m.ndim == 3:          # una sola caja: (1, H, W)
                    m = m[None]
                mascaras.append(m[:, 0] > 0.5)
        return np.concatenate(mascaras)


class Nombrador:
    def __init__(self):
        import torch
        from transformers import AutoModel, AutoProcessor

        self.torch = torch
        self.procesador = AutoProcessor.from_pretrained(SIGLIP)
        self.modelo = AutoModel.from_pretrained(SIGLIP, torch_dtype=torch.float16).to("cuda").eval()
        self.escala = float(self.modelo.logit_scale.exp().item())

    def _normalizar(self, v):
        v = v.float()
        return (v / v.norm(dim=-1, keepdim=True)).cpu().numpy()

    def vectores_imagen(self, imagenes):
        entradas = self.procesador(images=imagenes, return_tensors="pt").to("cuda")
        with self.torch.inference_mode():
            return self._normalizar(self.modelo.get_image_features(pixel_values=entradas.pixel_values.half()))

    def vectores_texto(self, textos):
        salida = []
        for inicio in range(0, len(textos), 64):
            entradas = self.procesador(text=textos[inicio:inicio + 64], padding="max_length", max_length=64,
                                       truncation=True, return_tensors="pt").to("cuda")
            with self.torch.inference_mode():
                salida.append(self._normalizar(self.modelo.get_text_features(**entradas)))
        return np.concatenate(salida)


class Modelos:
    """Carga perezosa: si no hay fotos útiles no se gasta tiempo en cargar los pesos."""

    def __init__(self):
        self._detector = self._segmentador = self._nombrador = None

    @property
    def detector(self):
        self._detector = self._detector or Detector()
        return self._detector

    @property
    def segmentador(self):
        self._segmentador = self._segmentador or Segmentador()
        return self._segmentador

    @property
    def nombrador(self):
        self._nombrador = self._nombrador or Nombrador()
        return self._nombrador

    versiones = ["grounding-dino-base", "sam2.1-hiera-large", "siglip2-so400m-patch14-384"]
