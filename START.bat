@echo off
chcp 65001 >nul 2>&1
title POE2 Price Overlay
color 0A


echo.
echo  POE2 Price Overlay
echo  Real-time item pricing
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
    python -m pip install -r "%~dp0requirements.txt" --quiet
    if errorlevel 1 (
        echo.
        echo   Failed to install dependencies.
        echo   Try manually: python -m pip install -r requirements.txt
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
    echo   1. Fate of the Vaal [current league]
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
echo  League: %LEAGUE%
echo  ──────────────────────────────
echo.
echo  Hover over items for prices
echo  Close window to stop
echo.

cd /d "%~dp0"
python src\main.py --league "%LEAGUE%"
