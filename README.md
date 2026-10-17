# Escáner LiDAR para iPhone 17 Pro

App iOS nativa (Swift, SwiftUI, ARKit y RealityKit) que combina el **sensor LiDAR**, la **cámara RGB** y el **seguimiento de movimiento** del iPhone para crear modelos 3D **a color y a escala real** (en metros).

## Modos

| Modo | Para qué | Cómo funciona | Resultado |
|---|---|---|---|
| **Entorno** | Habitaciones, casas, exteriores, muebles, objetos grandes | Malla de reconstrucción LiDAR de ARKit + mapa de profundidad por píxel + color de la cámara. Al terminar, cada vértice se colorea proyectando las fotos clave sobre él, con prueba de oclusión. | `malla_color.ply`, `malla_color.obj`, `malla_color.usdz`, `nube_puntos.ply` |
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

## Estructura

```
LidarScanner/
├── App/                     Entrada de la app y pantalla principal
├── Environment/             Modo Entorno
│   ├── SceneScanner.swift       Sesión ARKit: integra profundidad + color y exporta
│   ├── CameraModel.swift        Proyección y desproyección pinhole (intrínsecos de ARKit)
│   ├── ImageSampler.swift       YCbCr → RGB del fotograma de la cámara
│   ├── PointCloudAccumulator    Fusión de la nube de puntos en vóxeles (promedia ruido y color)
│   ├── KeyframeStore.swift      Selección de vistas bien repartidas para texturizar
│   ├── MeshBuilder.swift        Une los ARMeshAnchor en una sola malla en coordenadas del mundo
│   ├── MeshColorizer.swift      Color por vértice multivista con prueba de oclusión LiDAR
│   └── SceneScanView.swift      Interfaz del escaneo
├── ObjectCapture/           Modo Objeto (ObjectCaptureSession + PhotogrammetrySession)
└── Shared/                  Exportadores (PLY/OBJ/USDZ), almacenamiento, galería
```

## Calidad (modo Entorno)

| Calidad | Vóxel de la nube | Muestreo de profundidad | Uso |
|---|---|---|---|
| Rápido | 2 cm | 1 de cada 4 píxeles | Espacios grandes, vista previa |
| Equilibrado | 1 cm | 1 de cada 2 | Uso general |
| Preciso | 5 mm | todos | Muebles, detalle; consume más memoria |

Solo se usan mediciones LiDAR de **confianza alta**, entre 0,15 m y 4 m.

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

## Limitaciones conocidas y siguientes pasos

- En el modo Entorno el color es **por vértice**: la nitidez depende de la densidad de la malla de ARKit (unos 2–5 cm entre vértices). Para una textura más fina, el siguiente paso sería generar un **atlas UV de texturas** proyectando las fotos clave.
- La nube de puntos se procesa en la CPU; un paso futuro es moverla a **Metal** y mostrarla a color en tiempo real.
- No hay cierre de bucle ni alineamiento ICP propio: se confía en el seguimiento de ARKit, que es muy bueno en espacios del tamaño de una casa.
- La exportación a USDZ del modo Entorno usa SceneKit; si un visor no muestra el color de los vértices, usa el PLY u OBJ.
