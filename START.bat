@echo off
title POE2 Price Overlay
color 0A

echo.
echo   ╔══════════════════════════════════════════════╗
echo   ║        POE2 Price Overlay                    ║
echo   ║        Real-time item pricing                ║
echo   ╚══════════════════════════════════════════════╝
echo.

:: ─── Check Python ───────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo   ERROR: Python is not installed or not in PATH.
    echo.
    echo   Please install Python 3.10+ from:
    echo   https://www.python.org/downloads/
    echo.
    echo   IMPORTANT: Check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

for /f "tokens=2" %%a in ('python --version 2^>^&1') do set PYVER=%%a
echo   Python %PYVER% detected

:: ─── Check/Install Dependencies ─────────────────────
echo   Checking dependencies...

python -c "import requests" >nul 2>&1
if errorlevel 1 (
    echo   Installing required packages...
    pip install -r "%~dp0requirements.txt" --quiet
    if errorlevel 1 (
        echo.
        echo   Failed to install dependencies.
        echo   Try manually: pip install -r requirements.txt
        pause
        exit /b 1
    )
    echo   ✓ Dependencies installed
) else (
    echo   ✓ Dependencies OK
)

:: ─── Select League (first run) ──────────────────────
set LEAGUE_FILE=%USERPROFILE%\.poe2-price-overlay\league.txt

if not exist "%USERPROFILE%\.poe2-price-overlay" mkdir "%USERPROFILE%\.poe2-price-overlay"

if not exist "%LEAGUE_FILE%" (
    echo.
    echo   ────────────────────────────────────────────
    echo   Which league are you playing?
    echo   ────────────────────────────────────────────
    echo.
    echo   1. Fate of the Vaal (current temp league)
    echo   2. Standard
    echo   3. Hardcore Fate of the Vaal
    echo   4. Hardcore
    echo.
    set /p LEAGUE_CHOICE="  Enter number [1]: "

    if "%LEAGUE_CHOICE%"=="2" ( echo Standard>"%LEAGUE_FILE%" )
    if "%LEAGUE_CHOICE%"=="3" ( echo Hardcore Fate of the Vaal>"%LEAGUE_FILE%" )
    if "%LEAGUE_CHOICE%"=="4" ( echo Hardcore>"%LEAGUE_FILE%" )
    if not exist "%LEAGUE_FILE%" ( echo Fate of the Vaal>"%LEAGUE_FILE%" )
)

set /p LEAGUE=<"%LEAGUE_FILE%"

:: ─── Launch ─────────────────────────────────────────
color 0A
echo.
echo   ════════════════════════════════════════════
echo   Starting POE2 Price Overlay
echo   League: %LEAGUE%
echo   ════════════════════════════════════════════
echo.
echo   • Set POE2 to Windowed Fullscreen
echo   • Copy items with Ctrl+C in POE2 to get prices
echo   • Hover over items to see prices
echo   • Press Ctrl+C here to stop
echo.
echo   ────────────────────────────────────────────
echo.

cd /d "%~dp0"
python main.py --league "%LEAGUE%"

echo.
echo   Overlay stopped.
pause
