@echo off
title POE2 Price Overlay - Bug Report
color 0E

echo.
echo   ╔══════════════════════════════════════════════╗
echo   ║        POE2 Price Overlay - Bug Report        ║
echo   ╚══════════════════════════════════════════════╝
echo.

set DATA_DIR=%USERPROFILE%\.poe2-price-overlay
set ZIP_NAME=poe2-overlay-logs.zip
set ZIP_PATH=%USERPROFILE%\Desktop\%ZIP_NAME%

:: Check if data directory exists
if not exist "%DATA_DIR%" (
    echo   No log data found at %DATA_DIR%
    echo   Run the overlay at least once first.
    echo.
    pause
    exit /b 1
)

:: Delete old zip if it exists
if exist "%ZIP_PATH%" del "%ZIP_PATH%"

:: Zip the data directory
echo   Zipping logs and debug data...
powershell -Command "Compress-Archive -Path '%DATA_DIR%\*' -DestinationPath '%ZIP_PATH%' -Force" 2>nul

if exist "%ZIP_PATH%" (
    echo   ✓ Created: %ZIP_PATH%
) else (
    echo   Failed to create zip file.
    echo   You can manually zip the folder: %DATA_DIR%
    echo.
    pause
    exit /b 1
)

echo.
echo   ════════════════════════════════════════════
echo   Next steps:
echo   ════════════════════════════════════════════
echo.
echo   1. A GitHub issue page will open in your browser
echo   2. Describe what happened and what you expected
echo   3. Drag-and-drop the zip file from your Desktop:
echo      %ZIP_PATH%
echo   4. Submit the issue
echo.

pause

:: Open GitHub issue page
start https://github.com/CarbonSMASH/POE2_OCR/issues/new?template=bug_report.md
