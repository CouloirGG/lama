@echo off
REM LAMA Harvest Cycle - runs one harvest cycle (15 passes) with accuracy check.
REM Designed to be called by Windows Task Scheduler every 4-6 hours.
REM Uses a lock file to prevent overlapping runs.

setlocal
set LOCKFILE=%USERPROFILE%\.poe2-price-overlay\cache\harvest.lock
set LOGFILE=%USERPROFILE%\.poe2-price-overlay\cache\harvest_scheduler.log
set SRCDIR=C:\Users\Stu\GitHub\lama\src
set PYTHON=C:\Users\Stu\AppData\Local\Programs\Python\Python311\python.exe

REM Ensure cache dir exists
if not exist "%USERPROFILE%\.poe2-price-overlay\cache" mkdir "%USERPROFILE%\.poe2-price-overlay\cache"

REM Check for lock file (prevent overlapping runs)
if exist "%LOCKFILE%" (
    echo [%date% %time%] Skipping: previous harvest still running ^(lock file exists^) >> "%LOGFILE%"
    exit /b 0
)

REM Create lock file
echo %date% %time% > "%LOCKFILE%"

REM Run one harvest cycle with accuracy check
echo [%date% %time%] Starting harvest cycle >> "%LOGFILE%"
cd /d "%SRCDIR%"
"%PYTHON%" -u harvest_scheduler.py --once --passes 15 >> "%LOGFILE%" 2>&1
echo [%date% %time%] Harvest cycle complete (exit code: %ERRORLEVEL%) >> "%LOGFILE%"

REM Remove lock file
del "%LOCKFILE%" 2>nul

endlocal
