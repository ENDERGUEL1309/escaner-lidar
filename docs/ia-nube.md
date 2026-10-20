# Post-proceso con IA en la nube

Diseño del servicio que recibe un escaneo de entorno del iPhone, lo mejora con modelos de IA que necesitan GPU y devuelve una escena más completa y mejor etiquetada. Hay un prototipo de la API y del editor web en [`servidor/`](../servidor/README.md) y el trabajador con GPU (etapas A–D) en [`trabajador-gpu/`](../trabajador-gpu/README.md).

> Estado: **diseño + prototipo local + trabajador de GPU para Modal (sin desplegar)**. No hay nada desplegado ni cuentas creadas. Los precios y licencias son orientativos (octubre de 2026) y hay que confirmarlos antes de contratar.

## 1. Qué hace la nube que el iPhone no puede

| # | Función | Hoy en el iPhone | En la nube |
|---|---|---|---|
| 1 | **Segmentación 3D de vocabulario abierto** | 8 clases de ARKit + máscaras de instancia de Vision (`InstanceSignatures`) + `VNClassifyImageRequest` (~1300 etiquetas, foto a foto) | Detecta *cualquier* objeto por nombre ("lámpara", "planta", "monitor", "mando de la tele") con Grounding DINO / SAM 3, segmenta con SAM 2 en todas las fotos clave con identidades estables, y proyecta las máscaras sobre la malla votando por vértice (estilo OpenMask3D / SAI3D / Open3DIS). Separa instancias que se tocan (dos sillas juntas, un monitor sobre la mesa). |
| 2 | **Completado de geometría oculta** | Solo rellena huecos de hasta ~8 vóxeles prolongando superficie | Genera la parte trasera e inferior de cada objeto con un modelo de imagen→3D (SAM 3D Objects, TRELLIS…), la ajusta a la escala y posición medidas y **marca cada vértice inferido** para no confundirlo con lo medido. |
| 3 | **Reconstrucción fotorrealista (opcional)** | Textura de 12 MP horneada sobre la malla | Gaussian Splatting entrenado con las fotos clave y las poses de ARKit: reflejos, cristales y detalles finos que una malla no captura. Solo para ver, no para medir. |
| 4 | **Editor web** | `SceneViewer.swift` | three.js servido desde Railway: abre `escena.json` + `objetos/*.ply` + `textura.jpg`, mueve, gira y borra objetos y guarda el mismo `edicion.json` que el iPhone. |

## 2. Arquitectura

Railway **no tiene GPU**, así que se reparte en dos sitios:

- **Railway** (barato, siempre encendido): API, editor web, cola de trabajos y base de datos.
- **Proveedor de GPU sin servidor** (se paga por segundo de uso): **Modal** o **RunPod Serverless**. Arranca un contenedor con GPU solo cuando hay un trabajo.
- **Almacenamiento de objetos** compatible con S3: **Cloudflare R2** (sin coste de descarga) o el *bucket* de Railway. Los archivos pesados nunca pasan por la API: el iPhone sube y baja directamente con URLs prefirmadas.

```
┌──────────── iPhone ────────────┐
│ Escáner LiDAR                  │
│  1. POST /api/escaneos ────────┼──────────────┐
│  2. PUT archivos (URL prefirm.)┼───┐          │
│  3. POST /procesar ────────────┼───┼──────┐   │
│  4. GET estado / push APNs ◄───┼───┼──┐   │   │
│  5. GET resultados (URL pref.) ┼─┐ │  │   │   │
└────────────────────────────────┘ │ │  │   ▼   ▼
                                   │ │ ┌┴──────────────── Railway ─────────────────┐
                                   │ │ │ API (Node)  ·  Editor web (three.js)      │
                                   │ │ │ Postgres: escaneos, trabajos, dispositivos│
                                   │ │ │ Cola (tabla con SKIP LOCKED o Redis)      │
                                   │ │ └───────┬────────────────────▲──────────────┘
                                   │ │         │ lanza trabajo      │ progreso / fin
                                   │ │         │ (id + URLs pref.)  │ (webhook firmado HMAC)
                                   ▼ ▼         ▼                    │
                         ┌──── R2 / S3 ────┐  ┌──── Modal o RunPod (GPU) ──────────┐
                         │ entrada/<id>/…  │◄─┤ 1. descarga paquete                │
                         │ salida/<id>/…   │─►│ 2. segmentación 3D vocab. abierto  │
                         └─────────────────┘  │ 3. completado de geometría         │
                                              │ 4. Gaussian Splatting (opcional)   │
                                              │ 5. sube resultados                 │
                                              └────────────────────────────────────┘
```

### Flujo de un trabajo

1. **Crear** – la app envía el manifiesto (lista de archivos, tamaños, SHA-256 y opciones). La API crea el escaneo en estado `esperando_archivos` y devuelve una URL `PUT` prefirmada por archivo (caducan en 1 h).
2. **Subir** – la app sube cada archivo con una tarea de fondo de `URLSession` (sigue aunque la app se cierre). Se puede reintentar archivo a archivo.
3. **Procesar** – la app llama a `POST /procesar`. La API comprueba que están todos los archivos y encola el trabajo (`en_cola`).
4. **Ejecutar** – un despachador de Railway lanza la función de GPU pasándole el id y las URLs prefirmadas de lectura y escritura. Con Modal es `fn.spawn(...)`; con RunPod, `POST /v2/<endpoint>/run`. El trabajador informa del progreso a `POST /api/interno/trabajos/<id>/progreso` con una firma HMAC.
5. **Avisar** – al terminar, la API pasa el escaneo a `listo` y envía una notificación push (APNs). Sin push, la app consulta el estado cada 10–30 s mientras está abierta.
6. **Descargar** – la app baja los resultados a `Documentos/Escaneos/<carpeta>/ia/` y los integra (ver §6). El editor web los abre en `https://<app>.up.railway.app/editor/<id>`.

### Por qué así

- **Separar API y GPU** permite pagar la GPU solo por los minutos de cómputo; un servidor con GPU encendido 24 h cuesta más de 500 USD/mes.
- **Modal frente a RunPod.** Modal se programa en Python puro, sus arranques en frío son rápidos (imagen con los pesos en un volumen) y factura por segundo; es la opción recomendada para empezar. RunPod Serverless es algo más barato por hora, usa imágenes Docker propias y conviene si se quiere controlar el contenedor. El código del trabajador debe ser el mismo; solo cambia el envoltorio.
- **La cola en Postgres** (`SELECT … FOR UPDATE SKIP LOCKED`) basta para decenas de trabajos al día y evita otro servicio; Redis + BullMQ si se crece.
- **Idempotencia:** cada etapa escribe su salida en `salida/<id>/<etapa>/`; si el contenedor muere, al reintentar se saltan las etapas ya hechas.

## 3. Qué sube la app

Todo va dentro de la carpeta del escaneo con rutas relativas, para que el servidor la reconstruya tal cual.

| Archivo | Para qué | Tamaño típico | ¿Existe hoy? |
|---|---|---|---|
| `manifiesto.json` | Versión, opciones, lista de archivos con tamaño y SHA-256 | < 10 KB | **No** – nuevo |
| `fotos_clave/000.jpg` … | Fotos de 12 MP (4032×3024) de los fotogramas clave | 30–60 fotos × 3–5 MB | **No** – hoy se borran en `keyframes.reset()` |
| `fotos_clave/camaras.json` | Intrínsecos y pose de cada foto (ver abajo) | ~30 KB | **No** – nuevo |
| `fotos_clave/000.prof` … | Profundidad LiDAR 256×192 en Float16 + confianza (para la prueba de oclusión) | ~150 KB cada una | **No** – nuevo |
| `malla_color.ply` | Malla fusionada con normales y color por vértice | 20–80 MB | Sí |
| `escena.json` | Objetos detectados en el iPhone (punto de partida) | < 50 KB | Sí |
| `objetos/*.ply` | Malla de cada objeto con coordenadas de textura (`s`, `t`) | 20–80 MB en total | Sí |
| `textura.jpg` | Atlas de textura de 4096² u 8192² | 3–25 MB | Sí |
| `edicion.json` | Ediciones del usuario (para que el editor web las respete) | < 10 KB | Sí, si se editó |

**Volumen por escaneo:** 150–400 MB con fotos completas. Con la opción **ligera** (fotos reescaladas a 2048 px, JPEG 0,85, ~0,6 MB cada una) bajan a 60–150 MB, suficiente para la segmentación y el completado. Gaussian Splatting sí agradece la resolución completa. Por defecto: ligera con datos móviles, completa con Wi-Fi.

No se suben `modelo.obj/.usdz`, `nube_puntos.ply` ni `escena_objetos.*`: se derivan de lo anterior.

### `fotos_clave/camaras.json`

Convenciones iguales que en `CameraModel.swift`: intrínsecos en píxeles de *esa* foto (tipo OpenCV: x derecha, y abajo) y pose `cameraToWorld` de ARKit (x derecha, y arriba, z hacia atrás; metros; mundo con Y vertical). La matriz va **por columnas**, igual que `simd_float4x4`.

```json
{
  "version": 1,
  "convencion": "arkit",
  "fotos": [
    {
      "archivo": "000.jpg",
      "ancho": 4032, "alto": 3024,
      "fx": 2810.4, "fy": 2810.4, "cx": 2013.7, "cy": 1508.2,
      "cameraToWorld": [0.99, 0.0, -0.12, 0, 0.01, 0.99, 0.08, 0, 0.12, -0.08, 0.98, 0, 0.42, 1.31, -0.85, 1],
      "marcaTiempo": 12.533,
      "profundidad": { "archivo": "000.prof", "ancho": 256, "alto": 192,
                        "fx": 178.4, "fy": 178.4, "cx": 127.8, "cy": 95.7 }
    }
  ]
}
```

En el servidor se pasa a OpenCV (`world→cam` con flip de Y y Z) para COLMAP, gsplat y la proyección de máscaras.

### `manifiesto.json`

```json
{
  "version": 1,
  "nombre": "Entorno_2026-10-20_18-04-11",
  "app": "1.4 (iPhone18,1 · iOS 26.1)",
  "opciones": { "segmentacion": true, "completado": true, "splats": false, "vocabulario": ["lámpara", "planta"] },
  "archivos": [
    { "ruta": "malla_color.ply", "bytes": 48211002, "sha256": "…" },
    { "ruta": "fotos_clave/000.jpg", "bytes": 3820111, "sha256": "…" }
  ]
}
```

`vocabulario` es opcional: palabras que el usuario quiere que se busquen además de las automáticas.

## 4. Proceso en la GPU

```
fotos + poses + profundidad ─┐
malla_color.ply ─────────────┼─► A. Etiquetas por foto ─► B. Máscaras 2D con identidad ─► C. Elevación a 3D ─► D. Nombres
escena.json (iPhone) ────────┘        (RAM++/Florence-2)     (Grounding DINO o SAM 3        (votos por vértice,    (SigLIP 2 +
                                                              + SAM 2 vídeo)                 superpuntos)          vocabulario)
                                                                                                   │
                         G. Empaquetar ◄── F. Splats (opcional, gsplat) ◄── E. Completado por objeto ◄┘
                                                                            (SAM 3D Objects / TRELLIS + alineado ICP)
```

**A. Vocabulario.** Se juntan tres fuentes: etiquetas automáticas por foto (RAM++ o Florence-2), una lista fija de objetos de interior (las 200 clases de ScanNet200 traducidas) y las palabras del usuario.

**B. Máscaras 2D con identidad.** Grounding DINO (o SAM 3, que acepta conceptos en texto directamente) da cajas por palabra; SAM 2 convierte cada caja en máscara y, en modo vídeo sobre las fotos ordenadas por tiempo, **propaga la misma identidad** de una foto a la siguiente. Esto es lo que más mejora frente a Vision: dos sillas pegadas son dos pistas distintas en todas las vistas.

**C. Elevación a 3D.** Para cada vértice de la malla y cada foto: se proyecta con la pose, se descarta si la profundidad LiDAR dice que está tapado (|z − prof| > 3 cm) y se suma un voto a la máscara que lo cubre. La malla se divide antes en **superpuntos** (regiones pequeñas y planas, por normal y color) que se asignan en bloque, para que los bordes sean limpios. Los superpuntos se fusionan cuando la misma máscara los cubre en muchas vistas (como SAI3D / Open3DIS). Paredes, suelo y techo se toman de `escena.json` (ARKit ya los clasifica bien y la app los aplanó).

**D. Nombres.** De cada instancia se recortan sus 3–5 mejores vistas, se calcula el vector SigLIP 2 medio y se compara con el vocabulario. Se guarda la etiqueta, la confianza y el vector (permite buscar "¿dónde está algo para sentarse?" sin volver a procesar).

**E. Completado de geometría oculta.** Para cada objeto no estructural:
1. Se elige la vista más frontal y se recorta con su máscara.
2. Un modelo imagen→3D (**SAM 3D Objects**, o **TRELLIS** como alternativa) genera una malla completa.
3. Se alinea con lo medido: escala por la caja de `escena.json`, giro solo alrededor de Y (el objeto está apoyado) y ajuste fino ICP contra los vértices medidos.
4. **Fusión conservadora:** se conservan todos los vértices medidos. De la malla generada solo se añaden triángulos en zonas que el TSDF del iPhone nunca vio (detrás o debajo); lo que el LiDAR vio vacío no se rellena nunca.
5. Cada vértice añadido lleva `inferido = 1` y el objeto guarda `fraccionInferida`. Si el ICP no converge (error > 2 cm de media) el completado se descarta para ese objeto.

**F. Gaussian Splatting (opcional).** gsplat inicializado con la nube de puntos y las poses de ARKit (sin COLMAP), 7 000–15 000 iteraciones con ajuste fino de poses. Se exporta comprimido (`.spz`, 10–30 MB) para verlo en el editor web y en el iPhone.

**G. Empaquetar.** Ver formato de salida.

### Formato de salida (`salida/<id>/`)

| Archivo | Contenido |
|---|---|
| `escena.json` | **Versión 2**, compatible con la 1 (ver abajo) |
| `objetos/*.ply` | Malla por objeto con la nueva segmentación. Mismo formato que hoy (coordenadas de mundo, `s`/`t`, color) + propiedad de vértice `uchar inferido` |
| `textura.jpg` | La misma textura. Los vértices inferidos no tienen foto: llevan `s`/`t` apuntando a una zona gris neutra del atlas y el color estimado por vértice |
| `splats/escena.spz` | Solo si se pidió |
| `informe.json` | Tiempos, modelos y versiones usados, avisos |

`PLYReader.swift` ya localiza las propiedades por nombre y sabe saltar un `uchar` extra, así que el visor actual abre estos PLY sin cambios.

`escena.json` v2 añade campos opcionales a cada objeto; los que la app ya lee no cambian:

```json
{
  "version": 2, "unidades": "metros", "ejeVertical": "Y",
  "procesado": { "servicio": "escaner-lidar-ia", "fecha": "2026-10-20T18:20:00Z", "modelos": ["sam2.1-l", "grounding-dino-b", "siglip2-so400m", "sam3d-objects"] },
  "objetos": [
    {
      "nombre": "Lámpara 1", "categoria": "objeto", "estructura": false,
      "centro": [0.81, 0.62, -1.40], "dimensiones": [0.30, 1.24, 0.30], "rotacionY": 12.0,
      "vertices": 5210, "triangulos": 10044, "archivo": "objetos/Lampara_1.ply",
      "etiquetaIA": "lámpara de pie", "confianzaIA": 0.91,
      "fuente": "nube", "alternativas": [["lámpara", 0.91], ["perchero", 0.05]],
      "fraccionInferida": 0.27, "vistas": 14
    }
  ]
}
```

## 5. Costes por escaneo

Escaneo típico: una habitación, 40 fotos, 15 objetos. Precios de lista aproximados por segundo de GPU (Modal y RunPod rondan estas cifras; verificar):

| Etapa | GPU | Tiempo | Coste aprox. |
|---|---|---|---|
| Arranque en frío + descarga de datos | L4 / A10G (~0,80 USD/h) | 30–60 s | 0,01 USD |
| A–D Segmentación 3D | L4 / A10G | 2–4 min | 0,03–0,06 USD |
| E Completado (15 objetos × 10–20 s) | L40S / A100 40 GB (~2 USD/h) | 3–5 min | 0,10–0,17 USD |
| F Gaussian Splatting (opcional) | L40S / A100 | 8–15 min | 0,30–0,50 USD |
| **Total sin splats** | | **6–10 min** | **≈ 0,15–0,25 USD** |
| **Total con splats** | | **15–25 min** | **≈ 0,45–0,75 USD** |

Fijos al mes (orientativos):

| Concepto | Coste |
|---|---|
| Railway: API + Postgres (plan Hobby, uso bajo) | 5–10 USD |
| R2: 50 GB guardados (≈ 250 escaneos), descargas gratis | ≈ 0,75 USD |
| Modal: crédito gratuito mensual del plan inicial | cubre ~100–150 escaneos sin splats |

Para ahorrar: agrupar etapas en una sola GPU cuando el escaneo es pequeño, borrar las fotos de entrada a los 7 días (los resultados se quedan) y no pedir splats por defecto.

## 6. Modelos y licencias recomendadas

Lo importante: el **código original de 3D Gaussian Splatting (Inria) no permite uso comercial**; usar **gsplat**. Y **Hunyuan3D** excluye por licencia la UE, el Reino Unido y Corea del Sur, así que no sirve para usuarios en España.

| Uso | Recomendado | Licencia | Alternativas |
|---|---|---|---|
| Máscaras y seguimiento | **SAM 2.1** (Meta) | Apache 2.0 | — |
| Detección por texto | **Grounding DINO** (IDEA) / Grounded-SAM-2 | Apache 2.0 | **SAM 3** (licencia SAM de Meta, permite uso comercial con condiciones: revisar); OWLv2 (Apache 2.0) |
| Etiquetas automáticas | **RAM++** | Apache 2.0 | Florence-2 (MIT) |
| Nombres / búsqueda | **SigLIP 2** | Apache 2.0 | OpenCLIP (MIT; los pesos dependen del conjunto de datos) |
| Elevación 2D→3D | **Implementación propia** siguiendo OpenMask3D, SAI3D y Open3DIS | — | Los repos de investigación tienen licencias distintas; leer el código, no copiarlo sin revisar |
| Imagen→3D (completado) | **SAM 3D Objects** (Meta) | Licencia SAM de Meta (revisar) | **TRELLIS** (Microsoft, MIT); Stable Fast 3D (gratis por debajo de 1 M USD de ingresos); ~~Hunyuan3D~~ (excluye UE) |
| Gaussian Splatting | **gsplat** (nerfstudio) | Apache 2.0 | ~~Código de Inria~~ (no comercial) |
| Visor de splats web | **Spark** (sparkjs, three.js) | MIT | GaussianSplats3D (MIT) |
| Visor 3D web | **three.js** | MIT | — |

Antes de publicar: guardar en `informe.json` la versión exacta de cada modelo y revisar las condiciones de las licencias "SAM" de Meta, que cambian con cada versión.

## 7. Seguridad y privacidad

Los escaneos son el interior de casas: datos personales.

- **Autenticación:** cada iPhone se registra una vez y recibe un token de dispositivo (más adelante, Sign in with Apple). Toda la API exige `Authorization: Bearer`. El editor web abre un escaneo con un enlace firmado que caduca.
- **Nada público:** el *bucket* es privado; todo se sirve con URLs prefirmadas de corta duración.
- **Retención:** fotos de entrada borradas a los 7 días; resultados a los 90 días o cuando el usuario lo pida (`DELETE /api/escaneos/<id>`).
- **Proveedores:** elegir región de la UE en R2 y en el proveedor de GPU si se ofrece; Modal y RunPod no deben guardar nada fuera de la ejecución.
- **Webhooks internos** firmados con HMAC y un secreto compartido.
- **Límites:** tamaño máximo por archivo (500 MB) y por escaneo (1 GB), número de trabajos por día y dispositivo.

## 8. Qué habría que añadir en la app (Swift)

No se ha tocado el código Swift. Para conectar la app con este servicio:

1. **Conservar las fotos clave** – `SceneScanner.process` llama a `keyframes.reset()`, que borra las fotos de 12 MP del directorio temporal. Antes de eso, copiar `keyframe.fullImageURL` a `<carpeta>/fotos_clave/NNN.jpg` (opción "Guardar fotos para la IA en la nube" en `ScanSettings`, por el espacio que ocupa).
2. **Escribir `camaras.json`** – con `Keyframe.fullCamera` (`fx`, `fy`, `cx`, `cy`, `cameraToWorld`) y el tamaño `fullWidth`×`fullHeight`; añadir la marca de tiempo del `ARFrame` a `Keyframe`.
3. **Guardar la profundidad** – `Keyframe.depth` + `depthCamera` en Float16 (`.prof`: cabecera de 8 bytes ancho/alto + datos).
4. **Cliente de la API** – `CloudClient.swift` con `URLSession` en segundo plano (`URLSessionConfiguration.background`) para las subidas, reintentos por archivo y comprobación de SHA-256 (`CryptoKit`).
5. **Pantalla "Mejorar con IA en la nube"** – en la vista del escaneo: opciones (completado, splats, palabras extra), tamaño a subir, progreso por etapas y aviso de privacidad.
6. **Integrar resultados** – descargar a `<carpeta>/ia/` y, si el usuario acepta, sustituir `escena.json` y `objetos/` (guardando la versión del iPhone en `<carpeta>/original/`). Ampliar `SceneObjectInfo` con los campos opcionales de la v2 (`fuente`, `fraccionInferida`, `alternativas`).
7. **Mostrar lo inferido** – leer `inferido` en `PLYReader` y pintarlo en el visor con rayado o semitransparente, y excluirlo de las medidas.
8. **Notificaciones push** – registrar el token APNs en `POST /api/dispositivos`.
9. **ATS** – la API va por HTTPS (Railway lo da por defecto), sin excepciones en `Info.plist`.

## 9. Fases

| Fase | Contenido | Resultado |
|---|---|---|
| 0 (hecho) | Este documento + prototipo local con trabajador simulado + editor web | `servidor/` |
| 1 | Desplegar API y editor en Railway con almacenamiento en disco; subida manual desde el ordenador (carpeta del escaneo copiada desde Archivos) | Editor web accesible |
| 2 (código hecho, sin desplegar) | Trabajador real en Modal con etapas A–D (segmentación); modo de prueba con imágenes sintéticas mientras la app no guarda fotos | [`trabajador-gpu/`](../trabajador-gpu/README.md) |
| 3 | Cambios en la app (§8) | Flujo completo desde el iPhone |
| 4 | Etapa E (completado) y F (splats) | Objetos completos y vista fotorrealista |

## 10. Prototipo (`servidor/`)

Node 18+ sin dependencias externas. Implementa la API de §2 con almacenamiento en disco local en lugar de R2, un **trabajador simulado** en lugar de la GPU (recorre las etapas, copia los archivos y escribe `escena.json` v2 marcado como simulado) y el **editor web** con three.js. Instrucciones en [`servidor/README.md`](../servidor/README.md).
