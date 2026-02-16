@echo off
chcp 65001 >nul 2>&1
title POE2 Price Overlay [DEBUG]
color 0E


echo.
echo  POE2 Price Overlay [DEBUG MODE]
echo  Full diagnostic logging enabled
echo.

:: ─── Check Python ───────────────────────────────────
python --version >nul 2>&1
if errorlevel 1 (
    color 0C
    echo   ERROR: Python is not installed or not in PATH.
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
        pause
        exit /b 1
    )
    echo   ✓ Dependencies installed
) else (
    echo   ✓ Dependencies OK
)

:: ─── League ─────────────────────────────────────────
set LEAGUE_FILE=%USERPROFILE%\.poe2-price-overlay\league.txt

if not exist "%USERPROFILE%\.poe2-price-overlay" mkdir "%USERPROFILE%\.poe2-price-overlay"

if not exist "%LEAGUE_FILE%" (
    echo.
    echo   Using default league: Fate of the Vaal
    echo   Run START.bat first to select a different league.
    echo.
    set LEAGUE=Fate of the Vaal
) else (
    set /p LEAGUE=<"%LEAGUE_FILE%"
)

:: ─── Debug Info ─────────────────────────────────────
set LOG_DIR=%USERPROFILE%\.poe2-price-overlay
echo.
echo   League: %LEAGUE%
echo   Log file: %LOG_DIR%\overlay.log
echo   Debug clipboard saves: %LOG_DIR%\debug\
echo   ──────────────────────────────
echo.
echo   DEBUG features:
echo     - Verbose logging to console + log file
echo     - Clipboard text saved to debug\ folder
echo     - Mod classification details (key vs common)
echo     - Hybrid query diagnostics
echo     - Trade API query/response details
echo.
echo   Hover over items for prices
echo   Close window to stop
echo.

cd /d "%~dp0"
python main.py --league "%LEAGUE%" --debug

echo.
echo   Debug session ended.
echo   Log file: %LOG_DIR%\overlay.log
pause
