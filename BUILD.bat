@echo off
title POE2 Price Overlay - Build
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
echo [1/3] Installing dependencies...
pip install -r requirements.txt --quiet
pip install pyinstaller --quiet

:: Build with PyInstaller
echo [2/3] Building executable...
pyinstaller build.spec --noconfirm --clean

if errorlevel 1 (
    echo.
    echo BUILD FAILED. Check errors above.
    pause
    exit /b 1
)

:: Copy launcher
echo [3/3] Finalizing...
copy /Y launcher.py dist\POE2PriceOverlay\ >nul 2>&1

echo.
echo ================================================
echo   Build complete!
echo ================================================
echo.
echo   Output: dist\POE2PriceOverlay\
echo   Run:    dist\POE2PriceOverlay\POE2PriceOverlay.exe
echo.
echo   You can copy the POE2PriceOverlay folder
echo   anywhere and run it standalone.
echo.
pause
