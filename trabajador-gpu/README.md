# Trabajador con GPU (Modal)

Sustituye al trabajador simulado de [`servidor/src/trabajador.js`](../servidor/src/trabajador.js) por IA de verdad, siguiendo [`docs/ia-nube.md`](../docs/ia-nube.md) §2 y §4 (etapas **A–D**):

| Etapa | Qué hace | Modelo |
|---|---|---|
| A. Vocabulario | ~100 objetos de interior (español → inglés) + palabras del usuario + etiquetas del iPhone | — |
| B. Máscaras 2D | Cajas por palabra en cada vista y máscara por caja | Grounding DINO base + SAM 2.1 Large |
| C. Elevación a 3D | Proyecta cada vértice de `malla_color.ply` en cada vista, descarta los tapados (LiDAR o z-buffer de la malla) y vota por máscara; superpuntos de 5 cm que se fusionan cuando las vistas coinciden | Propio (estilo SAI3D / Open3DIS) |
| D. Nombres | Recorta las 5 mejores vistas de cada objeto y lo compara con el vocabulario | SigLIP 2 So400m |

Las etapas E (completar geometría oculta) y F (Gaussian Splatting) **no están hechas**: si se piden, se anotan en `informe.json` como omitidas.

> Estado: **código listo, sin desplegar**. No se ha creado ninguna cuenta. Probado en local con modelos falsos (ver [Pruebas sin GPU](#pruebas-sin-gpu)); los modelos reales no se han ejecutado todavía.

## Qué devuelve

En la misma forma que la app (para que el visor del iPhone y el editor web lo abran sin cambios):

- `escena.json` **versión 2**: la estructura (suelo, paredes, techo, puertas, ventanas) se copia del iPhone tal cual; el resto se resegmenta. Cada objeto nuevo lleva `fuente: "nube"`, `etiquetaIA`, `confianzaIA`, `alternativas`, `vistas` y `fraccionInferida: 0`. Lo que no se asigna a ningún objeto va a `Fragmentos`, como en la app.
- `objetos/*.ply`: PLY binario con `x, y, z, nx, ny, nz, s, t, red, green, blue` + `uchar inferido` (siempre 0 de momento) y `comment TextureFile ../textura.jpg`. Las coordenadas `s, t` se sacan de los `objetos/*.ply` del iPhone, así que la textura sigue cuadrando.
- `textura.jpg` (copia) e `informe.json` (vistas usadas, máscaras, tiempos por etapa, avisos).

## Sin fotos: modo de prueba

La app **todavía no guarda las fotos clave** (`keyframes.reset()` las borra; ver `docs/ia-nube.md` §8). Mientras tanto, si el escaneo no trae `fotos_clave/camaras.json`, el trabajador "fotografía" la malla coloreada desde 36 cámaras virtuales (desde el centro de la sala mirando alrededor y desde las esquinas) y usa esas imágenes como si fueran fotos, con su profundidad exacta. Funciona, pero detecta peor que con las fotos de 12 MP: los colores por vértice tienen mucha menos resolución.

Con fotos, lee `fotos_clave/camaras.json` (formato de `docs/ia-nube.md` §3: intrínsecos de la foto y `cameraToWorld` de ARKit por columnas) y, si las hay, las profundidades `.prof` (cabecera de 8 bytes ancho/alto + Float16, opcionalmente seguido de la confianza en `uint8`).

La opción `vistas` del manifiesto elige: `"auto"` (fotos si las hay), `"fotos"` (error si no las hay) o `"sinteticas"` (siempre el modo de prueba).

---

## 1. Instalar Python en Windows

Hace falta Python **3.10 o posterior** (Modal admite hasta la 3.15). Recomendado: 3.12.

1. Descarga el instalador de <https://www.python.org/downloads/windows/> ("Windows installer (64-bit)").
2. Al abrirlo, marca **"Add python.exe to PATH"** y pulsa *Install Now*.
3. Abre una terminal **nueva** (PowerShell) y comprueba:

```powershell
py --version
```

Debe decir `Python 3.12.x` (o la que hayas instalado). Si dice que no encuentra Python y abre la Microsoft Store, desactiva los alias en *Configuración › Aplicaciones › Configuración avanzada de aplicaciones › Alias de ejecución de aplicaciones* (`python.exe` y `python3.exe`) y abre otra terminal.

> En este PC parece haber ya un Python 3.14 del *Python Install Manager*; si `py --version` muestra 3.14, sirve tal cual.

Alternativa por terminal: `winget install -e --id Python.Python.3.12`.

## 2. Instalar Modal y entrar

Desde esta carpeta (`trabajador-gpu`):

```powershell
py -m pip install --upgrade modal
```

```powershell
py -m modal setup
```

`modal setup` abre el navegador para crear la cuenta o entrar (lo haces tú) y guarda el token en tu usuario. El plan gratuito incluye crédito mensual (ver `docs/ia-nube.md` §5).

## 3. Crear el secreto compartido

El servidor y la GPU firman sus mensajes con el mismo secreto. Genera uno largo:

```powershell
py -c "import secrets; print(secrets.token_hex(32))"
```

Y guárdalo en Modal (cambia `<secreto>` por lo que ha salido):

```powershell
py -m modal secret create escaner-lidar SECRETO_INTERNO=<secreto>
```

Apúntalo: el servidor necesitará el mismo valor en `SECRETO_INTERNO`.

## 4. Probar con un escaneo, sin servidor

Copia al PC la carpeta de un escaneo del iPhone (app **Archivos** › *En mi iPhone* › *Escáner LiDAR* › *Escaneos*) y:

```powershell
py -m modal run app.py --carpeta "C:\Users\User\Downloads\Entorno_2026-10-20_18-04-11"
```

- Sube `malla_color.ply`, `escena.json`, `objetos/*.ply`, `textura.jpg` y `fotos_clave/*` a un volumen de Modal, procesa en una GPU L4 y deja el resultado en `<carpeta>\ia\` (o en `--salida`).
- Opciones: `--sin-fotos` fuerza el modo de prueba; `--vocabulario "maceta,mando de la tele"` añade palabras.
- **La primera vez tarda bastante más** (≈10–15 min): Modal construye la imagen (PyTorch, SAM 2…) y descarga ~6 GB de pesos de Hugging Face a un volumen (`escaner-lidar-pesos`). Las siguientes arrancan en ~30–60 s.

Para ver el resultado, copia el contenido de `ia\` sobre la carpeta del escaneo (guarda antes una copia de `escena.json` y `objetos\`) o súbelo al editor web con `npm run subir` en `servidor/`.

## 5. Desplegar y conectar con el servidor

```powershell
py -m modal deploy app.py
```

Al terminar muestra la URL de la función web `lanzar`, del estilo `https://<usuario>--escaner-lidar-ia-lanzar.modal.run`. Después arranca el servidor (`servidor/`) con:

| Variable | Valor |
|---|---|
| `GPU_PROVEEDOR` | `modal` |
| `MODAL_URL` | La URL de `lanzar` |
| `URL_PUBLICA` | Dirección del servidor **accesible desde internet** (la GPU descarga de ahí y sube ahí): la de Railway (`https://<app>.up.railway.app`) o, para probar desde casa, un túnel como `cloudflared tunnel --url http://localhost:8080` |
| `SECRETO_INTERNO` | El mismo del paso 3 |

En PowerShell, para una prueba local con túnel:

```powershell
$env:GPU_PROVEEDOR="modal"; $env:MODAL_URL="https://…modal.run"; $env:URL_PUBLICA="https://…trycloudflare.com"; $env:SECRETO_INTERNO="<secreto>"; npm start
```

Sin `GPU_PROVEEDOR` (o con `simulado`) el servidor sigue usando el trabajador simulado y no necesita nada de esto.

### Cómo se comunican

1. `POST /api/escaneos/:id/procesar` → el servidor envía a `MODAL_URL` el trabajo firmado (`X-Firma` = HMAC-SHA256 del cuerpo): id, opciones, una URL firmada de descarga por archivo, una URL firmada de subida (`…/salida/{ruta}?expira&firma`) y la URL de progreso. Las URLs caducan a las 6 h (`VALIDEZ_URLS_S`).
2. `lanzar` comprueba la firma y la hora, lanza `procesar_trabajo` en la GPU (`spawn`) y responde enseguida.
3. La GPU descarga, procesa y sube; avisa de cada etapa en `POST /api/interno/trabajos/:id/progreso` (firmado igual) y al final manda `estado: "listo"` (o `"error"` con el mensaje).

Cuando los archivos vayan a R2/S3, el servidor solo tendrá que dar URLs prefirmadas de R2 en lugar de las suyas: el trabajador no cambia.

## Coste y GPU

Por defecto usa una **L4** (≈0,80 USD/h; un escaneo de una habitación sin fotos tarda unos minutos). Para otra GPU, define `ESCANER_GPU` antes de desplegar (p. ej. `$env:ESCANER_GPU="A10G"`). Límite de 45 min por trabajo. Nada se guarda en Modal salvo los pesos de los modelos; el volumen de pruebas se vacía al terminar cada `modal run`.

## Pruebas sin GPU

El proceso completo (todo menos los modelos) se puede probar en el PC con modelos falsos que "detectan" por color. Necesita `numpy`, `scipy` y `pillow`:

```powershell
py -m pip install numpy scipy pillow
```

```powershell
py pruebas\prueba_proceso.py
```

Crea una sala sintética con una caja roja (silla) pegada a una azul (mesa) que el iPhone habría dejado como un único objeto, la procesa en modo sin fotos y comprueba que salen «Silla 1» y «Mesa 1» separadas, con la estructura intacta y los PLY en el formato de la app.

Prueba del protocolo con el servidor, sin Modal: `pruebas\modal_falso.py` imita a `lanzar` en `http://localhost:8790` y ejecuta el trabajador real con los modelos falsos.

```powershell
$env:SECRETO_INTERNO="prueba"; py pruebas\modal_falso.py
```

Y en otra terminal, en `servidor/`:

```powershell
$env:GPU_PROVEEDOR="modal"; $env:MODAL_URL="http://localhost:8790"; $env:URL_PUBLICA="http://localhost:8080"; $env:SECRETO_INTERNO="prueba"; npm start
```

Luego `npm run subir -- <carpeta>` como siempre.

## Limitaciones conocidas

- **Modelos reales sin probar**: la integración con Grounding DINO, SAM 2 y SigLIP 2 (`escaner_ia/modelos.py`) sigue sus APIs de `transformers` 4.57 y `sam2` 1.1, pero no se ha ejecutado en una GPU. La primera ejecución real puede necesitar ajustes.
- **Identidad entre vistas en 3D, no con SAM 2 en modo vídeo**: las fotos clave están separadas en el tiempo, así que la identidad se resuelve fusionando superpuntos por coincidencia de máscaras entre vistas. Si las fotos pasan a ser casi continuas, el modo vídeo de SAM 2 podría añadirse en la etapa B.
- **Grounding DINO solo entiende inglés**: las palabras del usuario que no están en la tabla de `escaner_ia/vocabulario.py` se le pasan en español (SigLIP 2, que nombra, sí es multilingüe). Añadir traducciones a la tabla mejora la detección.
- **Parámetros** (`Parametros` en `escaner_ia/segmentacion.py`): tamaño de superpunto 5 cm, mínimo 300 vértices por objeto, 45 % de cobertura. Están ajustados a la escena sintética; habrá que revisarlos con escaneos reales.
- **Sin reanudación por etapas**: si el contenedor muere, el trabajo se repite entero (`docs/ia-nube.md` §2 propone guardar cada etapa).
- Sin GPU (pruebas), el rasterizado va en NumPy y es lento con mallas gruesas.

## Archivos

```
trabajador-gpu/
├── app.py                 App de Modal: imagen, `procesar_trabajo` (GPU), `lanzar` (web) y `modal run`
├── escaner_ia/
│   ├── trabajo.py         Un trabajo: descargar → procesar → subir → avisar
│   ├── pipeline.py        Etapas A–D y empaquetado (escena.json v2 + objetos/*.ply)
│   ├── segmentacion.py    B máscaras 2D, C elevación con oclusión y superpuntos, D nombres
│   ├── modelos.py         Grounding DINO, SAM 2, SigLIP 2
│   ├── vocabulario.py     Tabla de objetos español/inglés
│   ├── camaras.py         camaras.json, .prof, ARKit → OpenCV, cámaras virtuales
│   ├── render.py          Z-buffer por rasterizado (oclusión e imágenes sintéticas)
│   ├── ply.py             PLY binario con el formato de la app
│   └── comunicacion.py    Descargas, subidas y avisos firmados (solo biblioteca estándar)
└── pruebas/
    ├── falsos.py          Escena sintética y modelos falsos
    ├── prueba_proceso.py  Prueba del proceso completo
    └── modal_falso.py     Imita a Modal para probar el servidor en modo GPU_PROVEEDOR=modal
```
