@echo off
setlocal enabledelayedexpansion
chcp 65001 >nul 2>&1
title LAMA — Setup
cd /d "%~dp0"

:: ════════════════════════════════════════════════════════════
::   LAMA — One-Click Setup
:: ════════════════════════════════════════════════════════════

echo.
echo  ╔══════════════════════════════════════════════╗
echo  ║     LAMA — Setup                              ║
echo  ╚══════════════════════════════════════════════╝
echo.

:: ─── Verify script location ─────────────────────────────
if not exist "%~dp0requirements.txt" (
    echo   ERROR: SETUP.bat must be in the POE2_OCR folder.
    echo.
    echo   This file needs to be next to requirements.txt,
    echo   main.py, and the other project files.
    echo.
    echo   If you extracted from a ZIP, run SETUP.bat from
    echo   inside the extracted folder — don't copy it out.
    echo.
    pause
    exit /b 1
)

:: ─── Step 1: Find Python ──────────────────────────────────
echo  [1/4] Checking for Python...

set "PYTHON_CMD="

:: Try 'python' first
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :python_found
)

:: Try 'py' (Python Launcher — often available when python isn't on PATH)
py --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py"
    goto :python_found
)

:: Try common install locations
if exist "%LOCALAPPDATA%\Programs\Python\Python313\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python313\python.exe"
    goto :python_found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python312\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python312\python.exe"
    goto :python_found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python311\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python311\python.exe"
    goto :python_found
)
if exist "%LOCALAPPDATA%\Programs\Python\Python310\python.exe" (
    set "PYTHON_CMD=%LOCALAPPDATA%\Programs\Python\Python310\python.exe"
    goto :python_found
)

:: Python not found — offer to install
echo.
echo   Python is not installed on this computer.
echo.
echo   The installer will download Python from python.org
echo   and install it automatically.
echo.
set /p INSTALL_PY="  Install Python now? [Y/n]: "
if /i "!INSTALL_PY!"=="n" (
    echo.
    echo   Setup cancelled. Install Python 3.10+ manually from:
    echo   https://www.python.org/downloads/
    echo   IMPORTANT: Check "Add Python to PATH" during install.
    echo.
    pause
    exit /b 1
)

echo.
echo   Downloading Python installer...
set "PY_INSTALLER=%TEMP%\python-installer.exe"
powershell -Command "& { [Net.ServicePointManager]::SecurityProtocol = [Net.SecurityProtocolType]::Tls12; Invoke-WebRequest -Uri 'https://www.python.org/ftp/python/3.13.2/python-3.13.2-amd64.exe' -OutFile '%PY_INSTALLER%' }"
if errorlevel 1 (
    echo.
    echo   ERROR: Failed to download Python installer.
    echo   Please install Python 3.10+ manually from:
    echo   https://www.python.org/downloads/
    echo.
    pause
    exit /b 1
)

echo   Installing Python (this may take a minute)...
"%PY_INSTALLER%" /quiet InstallAllUsers=0 PrependPath=1 Include_launcher=1 Include_pip=1
if errorlevel 1 (
    echo.
    echo   ERROR: Python installation failed.
    echo   Try running the installer manually:
    echo   %PY_INSTALLER%
    echo.
    pause
    exit /b 1
)

:: Refresh PATH in current session
for /f "tokens=2*" %%a in ('reg query "HKCU\Environment" /v Path 2^>nul') do set "USER_PATH=%%b"
for /f "tokens=2*" %%a in ('reg query "HKLM\SYSTEM\CurrentControlSet\Control\Session Manager\Environment" /v Path 2^>nul') do set "SYS_PATH=%%b"
set "PATH=!USER_PATH!;!SYS_PATH!"

:: Verify Python is now available
python --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=python"
    goto :python_found
)
py --version >nul 2>&1
if not errorlevel 1 (
    set "PYTHON_CMD=py"
    goto :python_found
)

echo.
echo   ERROR: Python was installed but isn't available yet.
echo   Please close this window, open a NEW terminal, and
echo   run SETUP.bat again.
echo.
pause
exit /b 1

:python_found
for /f "tokens=*" %%v in ('!PYTHON_CMD! --version 2^>^&1') do set "PYVER=%%v"
echo        ✓ !PYVER! detected
echo.

:: ─── Step 2: Install Dependencies ─────────────────────────
echo  [2/4] Installing dependencies...

!PYTHON_CMD! -m pip install -r "%~dp0requirements.txt" --quiet --disable-pip-version-check 2>&1
if errorlevel 1 (
    echo.
    echo   Retrying with --user flag...
    !PYTHON_CMD! -m pip install -r "%~dp0requirements.txt" --quiet --user --disable-pip-version-check 2>&1
    if errorlevel 1 (
        echo.
        echo   ERROR: Failed to install dependencies.
        echo.
        echo   Try running manually:
        echo     !PYTHON_CMD! -m pip install -r requirements.txt
        echo.
        pause
        exit /b 1
    )
)
echo        ✓ Dependencies installed
echo.

:: ─── Step 3: Verify Installation ──────────────────────────
echo  [3/4] Verifying installation...

set "VERIFY_FAILED=0"
!PYTHON_CMD! -c "import webview" >nul 2>&1
if errorlevel 1 (
    echo        ✗ pywebview missing
    set "VERIFY_FAILED=1"
)
!PYTHON_CMD! -c "import fastapi" >nul 2>&1
if errorlevel 1 (
    echo        ✗ fastapi missing
    set "VERIFY_FAILED=1"
)
!PYTHON_CMD! -c "import requests" >nul 2>&1
if errorlevel 1 (
    echo        ✗ requests missing
    set "VERIFY_FAILED=1"
)

if "!VERIFY_FAILED!"=="1" (
    echo.
    echo   ERROR: Some packages failed to install properly.
    echo   Try running: !PYTHON_CMD! -m pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo        ✓ All packages verified
echo.

:: ─── Step 4: Desktop Shortcut ─────────────────────────────
echo  [4/4] Creating desktop shortcut...

set "SHORTCUT=%USERPROFILE%\Desktop\LAMA.lnk"
if not exist "!SHORTCUT!" (
    powershell -Command "& { $ws = New-Object -ComObject WScript.Shell; $s = $ws.CreateShortcut('%USERPROFILE%\Desktop\LAMA.lnk'); $s.TargetPath = '%~dp0LAMA.bat'; $s.WorkingDirectory = '%~dp0'; $s.IconLocation = 'shell32.dll,14'; $s.Description = 'LAMA — Live Auction Market Assessor'; $s.Save() }" >nul 2>&1
    if not errorlevel 1 (
        echo        ✓ Desktop shortcut created
    ) else (
        echo        - Shortcut creation skipped
    )
) else (
    echo        ✓ Desktop shortcut already exists
)
echo.

:: ─── Launch ───────────────────────────────────────────────
echo  ╔══════════════════════════════════════════════╗
echo  ║     Setup complete. Launching dashboard...   ║
echo  ╚══════════════════════════════════════════════╝
echo.

!PYTHON_CMD! "%~dp0src\app.py"

:: If dashboard exits with error, pause so user can read it
if errorlevel 1 (
    echo.
    echo   Dashboard exited with an error.
    pause
)
