@echo off
rem Compila e instala gsplat 1.5.3 desde el codigo fuente para la GPU de este PC (necesita CUDA 12.8 y Visual Studio 2022).
rem La rama principal de gsplat no compila con MSVC; la version 1.5.3 si, y compilada asi no usa la compilacion al vuelo
rem de PyTorch (que cambio en PyTorch 2.11).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%~dp0.venv-splat\Scripts;%CUDA_HOME%\bin;%PATH%"
set TORCH_CUDA_ARCH_LIST=12.0
set DISTUTILS_USE_SDK=1
set MAX_JOBS=16
set "FUENTE=%USERPROFILE%\Herramientas\gsplat"
if not exist "%FUENTE%" git clone --recursive --branch v1.5.3 https://github.com/nerfstudio-project/gsplat.git "%FUENTE%"
uv pip install --python "%~dp0.venv-splat\Scripts\python.exe" setuptools wheel ninja jaxtyping rich
uv pip install --python "%~dp0.venv-splat\Scripts\python.exe" --no-build-isolation --reinstall-package gsplat "%FUENTE%"
