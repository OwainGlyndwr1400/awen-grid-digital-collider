@echo off
title Awen Grid Digital Collider - Level I Run (144k nodes x 369 ticks)
rem Runs from this file's own folder, so double-clicking always works.
cd /d "%~dp0"

rem Pin the exact interpreter; pre-flight via the engine's own backend
rem (includes the user-site bootstrap for elevated windows).
set PY=C:\Python314\python.exe
%PY% -c "from awen_collider.engine import Backend; print('pre-flight:', Backend().describe())" || echo *** WARNING: pre-flight failed - run may fall back to CPU ***

%PY% -m awen_collider.run_collider --nodes 144000 --ticks 369 --collide-every 9 --json logs\level1_144k.json --html logs\level1_144k_report.html

echo.
echo Opening graphical run report + live dashboard...
start "" "logs\level1_144k_report.html"
start "" "visualizer.html"
echo.
pause
