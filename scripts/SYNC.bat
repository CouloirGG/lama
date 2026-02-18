@echo off
title POE2 Price Overlay - Sync
cd /d "%~dp0.."

echo ============================================================
echo   POE2 Price Overlay - Multi-Machine Sync
echo ============================================================
echo.

REM --- 1. Check git is available ---
where git >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: git is not installed or not in PATH
    echo Install from https://git-scm.com/downloads
    pause
    exit /b 1
)

REM --- 2. Check we're in a git repo ---
git rev-parse --git-dir >nul 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: Not a git repository. Clone first:
    echo   git clone https://github.com/CarbonSMASH/POE2_OCR.git
    pause
    exit /b 1
)

REM --- 3. Stash any local changes ---
echo [1/5] Checking local changes...
git diff --quiet --exit-code 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo   WARNING: You have uncommitted changes. Stashing them...
    git stash push -m "auto-stash before sync %date% %time%"
    echo   Stashed. Run 'git stash pop' to restore after sync.
) else (
    echo   Clean working tree.
)

REM --- 4. Pull latest from GitHub ---
echo.
echo [2/5] Pulling latest from GitHub...
git pull origin main
if %ERRORLEVEL% NEQ 0 (
    echo ERROR: git pull failed. Check your network connection.
    pause
    exit /b 1
)

REM --- 5. Install/update Python dependencies ---
echo.
echo [3/5] Installing Python dependencies...
python -m pip install -r requirements.txt --quiet 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: pip install had issues. Some features may not work.
    echo Try manually: pip install -r requirements.txt
)

REM --- 6. Verify key imports ---
echo.
echo [4/5] Verifying installation...
python -c "import webview; import fastapi; import uvicorn; import requests; print('  All dependencies OK')" 2>nul
if %ERRORLEVEL% NEQ 0 (
    echo WARNING: Some imports failed. Running full install...
    python -m pip install pywebview fastapi "uvicorn[standard]" websockets requests pywin32 pytest
)

REM --- 7. Show status ---
echo.
echo [5/5] Sync complete!
echo.
echo   Current commit:
git log --oneline -1
echo.
echo   Files: %CD%
echo.
echo   To run the dashboard:  python src\app.py
echo   Or double-click:       POE2 Dashboard.bat
echo.
echo ============================================================
pause
