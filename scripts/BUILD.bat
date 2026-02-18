@echo off
title POE2 Price Overlay - Build
cd /d "%~dp0.."
echo.
echo ================================================
echo   Building POE2 Price Overlay
echo ================================================
echo.

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

:: Install build dependencies
echo [1/2] Installing dependencies...
python -m pip install -r requirements.txt --quiet
python -m pip install pyinstaller --quiet

:: Build with PyInstaller
echo [2/2] Building executable...
python -m PyInstaller scripts\build.spec --noconfirm --clean

if errorlevel 1 (
    echo.
    echo BUILD FAILED. Check errors above.
    pause
    exit /b 1
)

echo.
echo ================================================
echo   Build complete!
echo ================================================
echo.
echo   Output: dist\POE2PriceOverlay\
echo   Run:    dist\POE2PriceOverlay\POE2PriceOverlay.exe
echo.
pause
