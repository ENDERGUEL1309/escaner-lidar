# Escáner LiDAR para iPhone 17 Pro

App iOS nativa (Swift, SwiftUI, ARKit y RealityKit) que combina el **sensor LiDAR**, la **cámara RGB** y el **seguimiento de movimiento** del iPhone para crear modelos 3D **a color y a escala real** (en metros).

## Modos

| Modo | Para qué | Cómo funciona | Resultado |
|---|---|---|---|
| **Entorno** | Habitaciones, casas, exteriores, muebles, objetos grandes | Fusión volumétrica (TSDF) de la profundidad LiDAR + color de la cámara, relleno de huecos y separación en objetos con la clasificación neuronal de ARKit. | Malla y nube de puntos a color (`.ply`, `.obj`, `.usdz`), escena separada por objetos y `escena.json` |
| **Objeto** | Objetos pequeños y medianos, personas quietas | Object Capture de Apple: captura guiada (el LiDAR delimita el objeto) + fotogrametría en el propio iPhone. | `modelo.usdz` con textura fotorrealista (PBR) |

El modo **Objeto** da el mejor resultado visual (textura de alta resolución). El modo **Entorno** cubre espacios grandes que la fotogrametría no puede manejar.

## Requisitos

- iPhone con LiDAR (iPhone 12 Pro o posterior; pensado para el iPhone 17 Pro) con iOS 18 o posterior.
- Un **Mac** con Xcode 16 o posterior. Las apps de iOS solo se compilan en macOS.
- Una cuenta de Apple (la gratuita sirve para instalarla en tu propio iPhone).

## Compilar e instalar

El proyecto se genera con [XcodeGen](https://github.com/yonaskolb/XcodeGen) a partir de `project.yml`:

```bash
brew install xcodegen
```

```bash
xcodegen generate
```

```bash
open LidarScanner.xcodeproj
```

En Xcode:

1. Selecciona el target **LidarScanner** › *Signing & Capabilities* › elige tu **Team**. Si hace falta, cambia el *Bundle Identifier* (`com.tuempresa.lidarscanner`) por uno único.
2. Conecta el iPhone y selecciónalo como destino. ARKit y LiDAR **no funcionan en el simulador**.
3. Pulsa ▶︎ Run. La primera vez, en el iPhone: *Ajustes › General › VPN y gestión de dispositivos* › confía en tu certificado, y activa el *Modo desarrollador* si te lo pide.

## Compilar e instalar sin Mac (desde Windows)

1. **Compilar en la nube:** sube esta carpeta a un repositorio de GitHub. Si es **público**, los Mac de GitHub Actions son gratis; si es privado, consume minutos de macOS. El workflow `.github/workflows/compilar-ipa.yml` genera el proyecto, compila la app sin firmar y publica `LidarScanner.ipa` como artefacto. Lo encuentras en la pestaña *Actions* › la ejecución › *Artifacts*.
2. **Firmar e instalar desde Windows:** instala [Sideloadly](https://sideloadly.io) y iTunes (versión web de Apple, no la de la Microsoft Store). Conecta el iPhone por cable, arrastra el `.ipa` a Sideloadly e inicia sesión con tu Apple ID; si quieres, usa uno secundario.
3. **En el iPhone:** *Ajustes › Privacidad y seguridad › Modo de desarrollador* › actívalo y reinicia. Luego, en *Ajustes › General › VPN y gestión de dispositivos*, confía en tu Apple ID.

Con un Apple ID gratuito la app **caduca a los 7 días**: vuelve a instalarla con Sideloadly cuando pase. Con la cuenta de Apple Developer (99 USD/año) dura 1 año y puede distribuirse por TestFlight.

## Cómo se procesa un escaneo de entorno

```
LiDAR (profundidad) ─┐
Cámara (color) ──────┼─► Fusión TSDF ─► Relleno de huecos ─► Superficie ─► Color multivista ─► Exportar
ARKit (pose) ────────┘                                         │
ARKit (clasificación neuronal: pared, suelo, mesa…) ───────────┴─► Segmentación en objetos ─► Escena editable
```

1. **Fusión volumétrica TSDF:** el espacio se divide en vóxeles que guardan la distancia a la superficie, promediada sobre todas las lecturas LiDAR. El ruido se cancela, los objetos que se movieron desaparecen y no quedan capas duplicadas.
2. **Relleno de huecos:** los vóxeles no observados que están rodeados de superficie toman el valor de sus vecinos. Así se cierran huecos de hasta unos 8 vóxeles. Solo prolonga superficies existentes; no inventa objetos.
3. **Superficie (Surface Nets):** se extrae una malla continua y suave de la isosuperficie.
4. **Color multivista:** cada vértice toma el color de las mejores fotos clave (frontales, cercanas y no tapadas).
5. **Segmentación:** cada vértice recibe la clase que la red neuronal de ARKit asignó en el iPhone. Suelo, paredes, techo, puertas y ventanas forman grupos. El resto se separa en objetos por conectividad (al quitar suelo y paredes, cada mueble queda aislado) y se nombra por su clase mayoritaria ("Mesa 1", "Asiento 2", "Objeto 3"…), con su caja orientada y sus medidas.

### Archivos de un escaneo de entorno

| Archivo | Contenido |
|---|---|
| `malla_color.ply` / `.obj` / `.usdz` | Superficie completa con color |
| `nube_puntos.ply` | Nube de puntos densa, sin ruido, con normales y color |
| `escena_objetos.obj` / `.usdz` | La escena separada: cada objeto es una pieza independiente con nombre |
| `escena.json` | Inventario: nombre, categoría, centro, medidas y giro de cada objeto |

En Blender, importa `escena_objetos.obj` (o el `.usdz`). Cada mueble aparece como un objeto propio que puedes mover, borrar, duplicar o reemplazar.

## Estructura

```
LidarScanner/
├── App/                     Entrada de la app y pantalla principal
├── Environment/             Modo Entorno
│   ├── SceneScanner.swift       Sesión ARKit, integración por fotograma y post-proceso
│   ├── TSDFVolume.swift         Fusión volumétrica, relleno de huecos y extracción (Surface Nets)
│   ├── SceneSegmenter.swift     Clasificación ARKit por vértice y separación en objetos
│   ├── CameraModel.swift        Proyección y desproyección pinhole (intrínsecos de ARKit)
│   ├── ImageSampler.swift       YCbCr → RGB del fotograma de la cámara
│   ├── KeyframeStore.swift      Selección de vistas bien repartidas para colorear
│   ├── MeshColorizer.swift      Color por vértice multivista con prueba de oclusión LiDAR
│   ├── MeshBuilder.swift        Malla de ARKit (respaldo si la fusión no tiene datos)
│   └── SceneScanView.swift      Interfaz del escaneo
├── ObjectCapture/           Modo Objeto (ObjectCaptureSession + PhotogrammetrySession)
└── Shared/                  Exportadores (PLY/OBJ/USDZ/JSON), almacenamiento, galería
```

## Calidad (modo Entorno)

| Calidad | Vóxel | Distancia máxima | Uso |
|---|---|---|---|
| Rápido | 2 cm | 4 m | Casas enteras, espacios grandes |
| Equilibrado | 1,5 cm | 4 m | Habitaciones, uso general |
| Preciso | 1 cm | 3 m | Muebles y detalle; consume más memoria |

Las lecturas LiDAR de confianza alta pesan más; las de confianza media cuentan menos y las de confianza baja se descartan. Si aparece "Memoria del escaneo llena", finaliza o usa una calidad menor.
## Consejos para la máxima precisión

- **Luz uniforme y abundante.** El color sale de la cámara: las sombras duras y los reflejos se quedan "pintados" en el modelo.
- **Muévete despacio** y a una distancia de 0,5 a 3 m. El LiDAR pierde precisión más allá de unos 4 o 5 m.
- **Solapa las pasadas.** Vuelve a zonas ya escaneadas para que ARKit corrija la deriva.
- Las superficies **negras, transparentes o espejadas** absorben o desvían el láser y salen con huecos.
- **Personas:** usa el modo Objeto; la persona debe quedarse completamente quieta mientras otra la rodea.
- En el modo Objeto, haz **2 o 3 vueltas** a distintas alturas y voltea el objeto si se puede, para capturar la base.

## Abrir los modelos

- **USDZ:** desde la app (botón AR), Archivos, Mensajes o Reality Composer.
- **PLY/OBJ con color por vértice:** Blender (en *Material*, usa un nodo *Color Attribute*), MeshLab o CloudCompare.
- Los archivos están en **Archivos › En mi iPhone › Escáner LiDAR › Escaneos**, y también se pueden compartir con AirDrop desde la app.

## Hoja de ruta de IA

Lo que ya funciona en el iPhone usa la red neuronal de clasificación de escena de ARKit. Los siguientes pasos necesitan modelos más grandes en un **servidor con GPU**. Railway puede alojar la web y la API, pero no tiene GPU, así que la parte de IA iría en un proveedor como RunPod o Modal.

| Paso | Qué haría | Técnica |
|---|---|---|
| Segmentación de vocabulario abierto | Reconocer cualquier objeto ("lámpara", "planta", "monitor"), no solo mesa/asiento | Máscaras 2D con SAM sobre las fotos clave, proyectadas y votadas en 3D (estilo OpenMask3D / SAM3D) |
| Separar objetos que se tocan | Distinguir un portátil sobre una mesa, sillas juntas | Segmentación de instancias 3D (Mask3D) |
| Completar geometría oculta | Reconstruir la parte trasera o inferior de los muebles | Completado de formas aprendido, marcado como "inferido" para no confundirlo con lo medido |
| Sustituir por modelos limpios | Cambiar un objeto escaneado por un modelo 3D equivalente y editable | Búsqueda por forma y categoría en bibliotecas de modelos |
| Editor web | Mover, borrar y reemplazar objetos desde Safari | Visor three.js alojado en Railway, que lee `escena.json` |

## Limitaciones conocidas

- La detección distingue las clases que reconoce ARKit (pared, suelo, techo, puerta, ventana, mesa, asiento). Lo demás aparece como "Objeto N".
- Los objetos que se tocan entre sí (por ejemplo, algo apoyado en una mesa) salen como un solo objeto.
- El relleno de huecos cierra huecos pequeños; las zonas grandes sin escanear siguen abiertas.
- No hay cierre de bucle propio: se confía en el seguimiento de ARKit, que es muy bueno en espacios del tamaño de una casa.
- La exportación a USDZ usa SceneKit; si un visor no muestra el color de los vértices, usa el PLY u OBJ.
