@echo off
title Awen Grid - Level I Cross-Scale Combined Report
cd /d "%~dp0"
set PY=C:\Python314\python.exe
%PY% "experiments\combine_reports.py"
start "" "logs\level1_combined_report.html"
pause
