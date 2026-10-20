# Servidor de IA en la nube (prototipo)

API y editor web del post-proceso con IA descrito en [`docs/ia-nube.md`](../docs/ia-nube.md). Es un **prototipo local**:

- La IA está **simulada**: el trabajador recorre las etapas (segmentación, completado, splats) y devuelve el resultado con el formato definitivo, pero los objetos son los que detectó el iPhone. El de verdad irá en Modal o RunPod con GPU.
- Los archivos se guardan en disco (`servidor/datos/`), no en R2/S3.
- El estado se guarda en archivos JSON, no en Postgres.

Solo necesita **Node.js 18 o posterior**. No tiene dependencias que instalar (el editor carga three.js desde jsDelivr).

## Arrancar

```bash
cd servidor
```

```bash
npm start
```

Abre <http://localhost:8080>. En otra terminal, crea y sube un escaneo de prueba (habitación con mesa, silla y lámpara):

```bash
npm run demo
```

Al terminar muestra el enlace al editor web (`http://localhost:8080/editor/<id>`).

### Abrir un escaneo real del iPhone

1. En el iPhone, app **Archivos** › *En mi iPhone* › *Escáner LiDAR* › *Escaneos*: copia la carpeta del escaneo al ordenador (AirDrop no existe en Windows: usa iCloud Drive, OneDrive o un cable con iTunes).
2. Súbela:

```bash
npm run subir -- "C:\Users\User\Downloads\Entorno_2026-10-20_18-04-11"
```

Se suben `malla_color.ply`, `escena.json`, `objetos/*.ply`, `textura.jpg`, `edicion.json` y, cuando la app las guarde, `fotos_clave/*`.

### Variables de entorno

| Variable | Por defecto | Para qué |
|---|---|---|
| `PORT` | `8080` | Puerto (Railway lo define solo) |
| `DATOS` | `./datos` | Carpeta de datos. En Railway, un volumen montado |
| `API_TOKEN` | — | Si se define, toda la API exige `Authorization: Bearer <token>` (o `?token=` en el editor) |
| `SECRETO_INTERNO` | — | Secreto HMAC para los avisos de progreso del trabajador con GPU |
| `PAUSA_SIMULADA_MS` | `800` | Duración de cada etapa simulada |
| `SERVIDOR` | `http://localhost:8080` | A qué servidor suben `npm run demo` y `npm run subir` |

## API

Todas las respuestas son JSON. Con `API_TOKEN`, añadir `Authorization: Bearer <token>`.

| Método y ruta | Qué hace |
|---|---|
| `GET /api/salud` | Comprobación de vida |
| `POST /api/escaneos` | Crea un escaneo a partir del manifiesto. Devuelve `id` y una URL de subida por archivo |
| `PUT /api/escaneos/:id/archivos/<ruta>` | Sube un archivo (cuerpo binario). Comprueba tamaño y SHA-256 |
| `POST /api/escaneos/:id/procesar` | Encola el procesado (409 si faltan archivos) |
| `GET /api/escaneos/:id` | Estado: `esperando_archivos`, `en_cola`, `procesando`, `listo` o `error`, con `progreso` (0–1), `etapa` y `resultados` |
| `GET /api/escaneos/:id/resultado/<ruta>` | Descarga un resultado (`escena.json`, `objetos/*.ply`, `informe.json`…) |
| `GET /api/escaneos/:id/archivos/<ruta>` | Descarga un archivo subido |
| `GET`/`PUT /api/escaneos/:id/edicion` | Ediciones del editor web, mismo formato que `edicion.json` del iPhone |
| `GET /api/escaneos` | Lista de escaneos |
| `DELETE /api/escaneos/:id` | Borra el escaneo y todos sus archivos |
| `POST /api/interno/trabajos/:id/progreso` | Aviso del trabajador con GPU, firmado con HMAC-SHA256 del cuerpo en la cabecera `X-Firma` |
| `GET /editor/:id` | Editor web |

Ejemplo de manifiesto:

```json
{
  "version": 1,
  "nombre": "Entorno_2026-10-20_18-04-11",
  "opciones": { "segmentacion": true, "completado": true, "splats": false, "vocabulario": ["planta"] },
  "archivos": [
    { "ruta": "malla_color.ply", "bytes": 48211002, "sha256": "…" },
    { "ruta": "escena.json", "bytes": 4210 }
  ]
}
```

Límites: 500 MB por archivo, 1 GB por escaneo, 2000 archivos.

## Editor web

`/editor/<id>` hace lo mismo que el visor del iPhone (`SceneViewer.swift`):

- Clic en un objeto (o en la lista) para seleccionarlo; los demás se atenúan.
- **Mover** (flechas en el plano del suelo), **Girar** (anillo vertical), **↑ / ↓** 2 cm, **Aislar**, **Borrar**.
- Lista de objetos con ocultar/mostrar y sección *Borrados* con *Recuperar*.
- **Deshacer** (también Ctrl+Z), **Restaurar original**, **Sombreado** para ver el relieve.
- El techo empieza oculto.

Cada cambio se guarda en `edicion.json` con el mismo formato que la app (`dx`, `dy`, `dz`, `yaw` en radianes, `borrado`), así que podría volver al iPhone. Si el escaneo ya se procesó, el editor abre el resultado de la nube; si no, los archivos subidos.

## Desplegar en Railway (cuando toque)

No se ha desplegado nada. Los pasos serían:

1. Nuevo servicio desde el repositorio de GitHub con *Root Directory* `servidor` (Railway detecta Node y ejecuta `npm start`).
2. Añadir un **volumen** montado en `/datos` y la variable `DATOS=/datos`.
3. Definir `API_TOKEN` y `SECRETO_INTERNO`.
4. Más adelante: Postgres para el estado, R2 para los archivos y el trabajador real en Modal (ver `docs/ia-nube.md` §2 y §9).

## Archivos

```
servidor/
├── src/
│   ├── servidor.js       API HTTP, validación y rutas
│   ├── almacen.js        Escaneos en disco (sustituible por Postgres + R2)
│   └── trabajador.js     Trabajador simulado y cola en memoria
├── public/
│   ├── index.html        Lista de escaneos
│   ├── editor.html/.js   Editor 3D (three.js)
│   └── estilo.css
└── scripts/
    ├── cliente.js        Cliente de la API (lo que hará la app)
    ├── demo.js           Escaneo sintético de prueba
    └── subir-carpeta.js  Sube la carpeta de un escaneo real
```
