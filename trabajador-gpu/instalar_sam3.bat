@echo off
rem SAM3 (segmentacion de cada foto por conceptos) en su propio entorno: necesita transformers 5, que no encaja con
rem el de Hunyuan3D-2. Lo usa el remodelado (segmentar_fotos.py) si existe %USERPROFILE%\Herramientas\SAM3\.venv.
rem El modelo facebook/sam3 es restringido: acepta la licencia en https://huggingface.co/facebook/sam3 con tu cuenta
rem y despues inicia sesion aqui con un token de lectura (te lo pide este script).
setlocal
set "DESTINO=%USERPROFILE%\Herramientas\SAM3"
if not exist "%DESTINO%" mkdir "%DESTINO%"
cd /d "%DESTINO%"
if not exist ".venv\Scripts\python.exe" uv venv --python 3.12 .venv || goto :error
uv pip install --python .venv\Scripts\python.exe torch torchvision --index-url https://download.pytorch.org/whl/cu128 || goto :error
uv pip install --python .venv\Scripts\python.exe "transformers>=5" accelerate pillow numpy huggingface_hub || goto :error
.venv\Scripts\hf.exe auth whoami >nul 2>&1 || .venv\Scripts\hf.exe auth login
.venv\Scripts\python.exe -c "from huggingface_hub import hf_hub_download as d; d('facebook/sam3', 'config.json'); print('Acceso a SAM3: OK')" || echo Sin acceso a facebook/sam3 todavia: acepta la licencia en Hugging Face y vuelve a ejecutar este script.
echo Listo.
exit /b 0
:error
echo Fallo la instalacion.
exit /b 1
