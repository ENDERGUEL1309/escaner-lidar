@echo off
rem Procesa un escaneo completo en este PC (fotogrametria, 2DGS, malla, planos y version ligera).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%~dp0.venv-splat\Scripts;%CUDA_HOME%\bin;%PATH%"
set TORCH_CUDA_ARCH_LIST=12.0
set DISTUTILS_USE_SDK=1
set PYTHONIOENCODING=utf-8
"%~dp0.venv-splat\Scripts\python.exe" "%~dp0procesar_escaneo.py" %*
