@echo off
title POE2 Price Overlay - Diagnostics
color 0F

echo.
echo   Running diagnostics...
echo   Keep POE2 open in Windowed Fullscreen.
echo.

cd /d "%~dp0"
python diagnose.py
