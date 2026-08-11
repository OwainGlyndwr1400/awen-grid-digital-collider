@echo off
title Awen Grid Digital Collider - Level I Run (10M nodes x 369 ticks)
rem Runs from this file's own folder, so double-clicking always works.
cd /d "%~dp0"

rem Pin the exact interpreter; pre-flight via the engine's own backend
rem (includes the user-site bootstrap for elevated windows).
set PY=C:\Python314\python.exe
%PY% -c "from awen_collider.engine import Backend; print('pre-flight:', Backend().describe())" || echo *** WARNING: pre-flight failed - run may fall back to CPU ***

echo.
echo 10,000,000 nodes per ledger - expect roughly 2-5 minutes.
echo The 41 full-beam collision measurements dominate the runtime.
echo.

%PY% -m awen_collider.run_collider --nodes 10000000 --ticks 369 --collide-every 9 --json logs\level1_10M.json --html logs\level1_10M_report.html

echo.
echo Opening graphical run report + live dashboard...
start "" "logs\level1_10M_report.html"
start "" "visualizer.html"
echo.
pause
