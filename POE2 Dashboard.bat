@echo off
title POE2 Price Overlay Dashboard
cd /d "%~dp0"

:: Quick dep check — self-heals if deps are missing
python -c "import webview" >nul 2>&1
if errorlevel 1 (
    echo   Installing dependencies...
    python -m pip install -r "%~dp0requirements.txt" --quiet --disable-pip-version-check
)

python src\app.py
if errorlevel 1 pause
