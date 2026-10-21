@echo off
rem Ejecuta splat2d_local.py con el compilador de Visual Studio y CUDA 12.8 (gsplat compila sus kernels la primera vez).
call "C:\Program Files\Microsoft Visual Studio\2022\Community\VC\Auxiliary\Build\vcvars64.bat" >nul
set "CUDA_HOME=C:\Program Files\NVIDIA GPU Computing Toolkit\CUDA\v12.8"
set "PATH=%~dp0.venv-splat\Scripts;%CUDA_HOME%\bin;%PATH%"
set TORCH_CUDA_ARCH_LIST=12.0
set DISTUTILS_USE_SDK=1
"%~dp0.venv-splat\Scripts\python.exe" "%~dp0splat2d_local.py" %*
