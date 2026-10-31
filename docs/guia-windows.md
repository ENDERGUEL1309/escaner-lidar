# Guía en Windows (PowerShell)

Todos los comandos se ejecutan en PowerShell, desde la carpeta del repositorio salvo que se indique otra cosa.

## 1. Requisitos (una sola vez)

```powershell
winget install --id GitHub.cli -e
```

Cierra y abre PowerShell, y luego inicia sesión en GitHub:

```powershell
gh auth login
```

Node.js 22.2 o posterior ya basta para el servidor: no tiene dependencias que instalar.

## 2. Subir el código y compilar el `.ipa`

El workflow se lanza solo al subir a `main` o a cualquier rama `feat/...`.

```powershell
git add -A
git commit -m "Describe tu cambio"
git push -u origin HEAD
```

Para lanzarlo a mano en la rama actual:

```powershell
gh workflow run compilar-ipa.yml --ref (git branch --show-current)
```

## 3. Seguir la compilación y ver los errores

```powershell
gh run list --workflow compilar-ipa.yml --limit 5
```

```powershell
gh run watch
```

Si falla, muestra solo los pasos con error:

```powershell
gh run view --log-failed
```

El workflow también guarda el registro completo:

```powershell
gh run download --name log-compilacion
```

## 4. Instalar en el iPhone con Sideloadly

Descarga el artefacto de la última compilación:

```powershell
gh run download --name LidarScanner-ipa --dir ipa
```

Arrastra `ipa\LidarScanner.ipa` a Sideloadly con el iPhone conectado por cable. Con un Apple ID gratuito la app caduca a los 7 días.

## 5. Arrancar el servidor de refinado

```powershell
cd pc-windows-server
```

```powershell
npm start
```

Muestra la dirección y el token que hay que escribir en el iPhone. El escaneo y los resultados quedan en `pc-windows-server\datos\` hasta que el iPhone los descarga.

La primera vez, abre el puerto en el Firewall desde un PowerShell **como administrador**:

```powershell
New-NetFirewallRule -DisplayName "Escaner LiDAR refinado" -Direction Inbound -Protocol TCP -LocalPort 8787 -Action Allow -Profile Private
```

Tu Wi-Fi debe estar marcada como red **privada** en Windows. Comprueba el perfil con:

```powershell
Get-NetConnectionProfile
```

## 6. Usar «Mejorar en PC»

1. En la app, abre un escaneo de entorno hecho con «Guardar fotos para IA en la nube» activado.
2. En la sección **Mejorar en PC**, toca **Dirección y token del PC**, escribe los datos que mostró el servidor y prueba la conexión.
3. Toca **Mejorar en PC (Wi-Fi)**. iOS pedirá permiso de red local la primera vez: acéptalo.

Deja la pantalla abierta: la app consulta el estado y descarga el resultado mientras está ahí. Si sales, el PC sigue trabajando y al volver se descarga.

## 7. Fotogrametría completa (opcional)

La opción **Fotogrametría completa** usa `trabajador-gpu/procesar_escaneo.py` (COLMAP + 2DGS). Necesita el entorno `trabajador-gpu\.venv-splat`, o bien la variable `ESCANER_PYTHON` con la ruta de ese Python, además de GPU. Si no está instalado, el servidor entrega el refinado LiDAR y lo anota en `informe.json`.

## 8. Comprobar el servidor

```powershell
cd pc-windows-server
```

```powershell
npm test
```

## Qué hace y qué no hace cada refinado

| | Refinado LiDAR (siempre) | Fotogrametría completa |
|---|---|---|
| Alineación ICP densa de los mapas LiDAR | Sí | Sí |
| Ajuste de haces fotogramétrico | No | Sí (COLMAP) |
| OBJ, GLB, USDZ, plano DXF | Sí | Sí |
| Textura | Color por vértice | Color por vértice |

El servidor no genera una textura sin costuras: el `.ply` que envía el iPhone no lleva coordenadas de textura.
