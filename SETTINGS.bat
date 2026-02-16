@echo off
title POE2 Price Overlay - Settings
color 0B

set LEAGUE_FILE=%USERPROFILE%\.poe2-price-overlay\league.txt
set SETUP_FILE=%USERPROFILE%\.poe2-price-overlay\.setup_complete

echo.
echo   ╔══════════════════════════════════════════════╗
echo   ║        POE2 Price Overlay - Settings          ║
echo   ╚══════════════════════════════════════════════╝
echo.

:: Show current league
if exist "%LEAGUE_FILE%" (
    set /p CURRENT_LEAGUE=<"%LEAGUE_FILE%"
    echo   Current league: %CURRENT_LEAGUE%
) else (
    echo   Current league: Not set
)
echo.

:: Menu
echo   1. Change league
echo   2. Reset first-run setup
echo   3. Open log file
echo   4. Open cache folder
echo   5. Run pipeline tests
echo   0. Exit
echo.
set /p CHOICE="  Enter choice: "

if "%CHOICE%"=="1" goto :change_league
if "%CHOICE%"=="2" goto :reset_setup
if "%CHOICE%"=="3" goto :open_log
if "%CHOICE%"=="4" goto :open_cache
if "%CHOICE%"=="5" goto :run_tests
goto :eof

:change_league
echo.
echo   Select league:
echo   1. Fate of the Vaal (current temp league)
echo   2. Standard
echo   3. Hardcore Fate of the Vaal
echo   4. Hardcore
echo.
set /p LC="  Enter number: "

if not exist "%USERPROFILE%\.poe2-price-overlay" mkdir "%USERPROFILE%\.poe2-price-overlay"

if "%LC%"=="1" ( echo Fate of the Vaal>"%LEAGUE_FILE%" & echo   Set to Fate of the Vaal )
if "%LC%"=="2" ( echo Standard>"%LEAGUE_FILE%" & echo   Set to Standard )
if "%LC%"=="3" ( echo Hardcore Fate of the Vaal>"%LEAGUE_FILE%" & echo   Set to Hardcore Fate of the Vaal )
if "%LC%"=="4" ( echo Hardcore>"%LEAGUE_FILE%" & echo   Set to Hardcore )
echo.
pause
goto :eof

:reset_setup
if exist "%SETUP_FILE%" del "%SETUP_FILE%"
echo   Setup reset. Next launch will run first-time setup.
pause
goto :eof

:open_log
set LOG=%USERPROFILE%\.poe2-price-overlay\overlay.log
if exist "%LOG%" (
    notepad "%LOG%"
) else (
    echo   No log file found. Run the overlay first.
    pause
)
goto :eof

:open_cache
set CACHE=%USERPROFILE%\.poe2-price-overlay\cache
if exist "%CACHE%" (
    explorer "%CACHE%"
) else (
    echo   No cache folder found. Run the overlay first.
    pause
)
goto :eof

:run_tests
echo.
cd /d "%~dp0"
python test_pipeline.py
echo.
pause
goto :eof