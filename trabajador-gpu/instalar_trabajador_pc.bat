@echo off
rem Hace que el trabajador del PC arranque (minimizado) al iniciar sesion en Windows. Para quitarlo, borra
rem "Escaner LiDAR - trabajador.lnk" de la carpeta Inicio (Win+R, shell:startup).
powershell -NoProfile -Command "$s=(New-Object -ComObject WScript.Shell).CreateShortcut([Environment]::GetFolderPath('Startup')+'\Escaner LiDAR - trabajador.lnk'); $s.TargetPath='%~dp0trabajador_pc.bat'; $s.WorkingDirectory='%~dp0'; $s.WindowStyle=7; $s.Save()"
echo Listo: el trabajador arrancara con Windows. Para lanzarlo ahora: trabajador_pc.bat
