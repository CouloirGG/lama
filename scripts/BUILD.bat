@echo off
title POE2 Price Overlay - Build
cd /d "%~dp0.."
echo.
echo ================================================
echo   Building POE2 Price Overlay
echo ================================================
echo.

:: Read version from resources\VERSION
set /p APP_VERSION=<resources\VERSION

:: Check Python
python --version >nul 2>&1
if errorlevel 1 (
    echo ERROR: Python not found. Install Python 3.10+ from python.org
    pause
    exit /b 1
)

:: Install build dependencies
echo [1/3] Installing dependencies...
python -m pip install -r requirements.txt --quiet
python -m pip install pyinstaller --quiet

:: Build with PyInstaller
echo [2/3] Building executable...
python -m PyInstaller scripts\build.spec --noconfirm --clean

if errorlevel 1 (
    echo.
    echo BUILD FAILED. Check errors above.
    pause
    exit /b 1
)

:: Build installer with Inno Setup (optional)
echo [3/3] Building installer...
set "ISCC_EXE="

:: Check PATH first
where ISCC.exe >nul 2>&1
if not errorlevel 1 (
    set "ISCC_EXE=ISCC.exe"
    goto :run_iscc
)

:: Check default install locations
if exist "C:\Program Files (x86)\Inno Setup 6\ISCC.exe" (
    set "ISCC_EXE=C:\Program Files (x86)\Inno Setup 6\ISCC.exe"
    goto :run_iscc
)
if exist "C:\Program Files\Inno Setup 6\ISCC.exe" (
    set "ISCC_EXE=C:\Program Files\Inno Setup 6\ISCC.exe"
    goto :run_iscc
)

echo   Inno Setup not found — skipping installer build.
echo   Install from https://jrsoftware.org/isinfo.php to build the Setup exe.
goto :done

:run_iscc
echo   Using: %ISCC_EXE%
"%ISCC_EXE%" scripts\installer.iss

if errorlevel 1 (
    echo.
    echo   WARNING: Installer build failed. The portable exe is still available.
) else (
    echo   Installer: dist\POE2PriceOverlay-Setup-%APP_VERSION%.exe
)

:done
echo.
echo ================================================
echo   Build complete!
echo ================================================
echo.
echo   Portable:  dist\POE2PriceOverlay\POE2PriceOverlay.exe
if defined ISCC_EXE echo   Installer: dist\POE2PriceOverlay-Setup-%APP_VERSION%.exe
echo.
pause
