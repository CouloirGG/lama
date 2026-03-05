@echo off
REM LAMA Disappearance Tracker - checks if harvested listings have sold or gone stale.
REM Designed to run independently from harvest_cycle.bat via Windows Task Scheduler.
REM Runs every 6 hours, checking up to 2000 listing IDs per run.

setlocal EnableDelayedExpansion
set LOCKFILE=%USERPROFILE%\.poe2-price-overlay\cache\disappearance.lock
set LOGFILE=%USERPROFILE%\.poe2-price-overlay\cache\disappearance.log
set SRCDIR=C:\Users\Stu\GitHub\lama\src
set PYTHON=C:\Users\Stu\AppData\Local\Programs\Python\Python311\python.exe

REM Ensure cache dir exists
if not exist "%USERPROFILE%\.poe2-price-overlay\cache" mkdir "%USERPROFILE%\.poe2-price-overlay\cache"

REM Check for lock file (prevent overlapping runs)
if exist "%LOCKFILE%" (
    for /f %%A in ('powershell -NoProfile -Command "(New-TimeSpan -Start (Get-Item \"%LOCKFILE%\").LastWriteTime -End (Get-Date)).TotalHours"') do set LOCK_AGE_H=%%A
    for /f "tokens=1 delims=." %%I in ("!LOCK_AGE_H!") do set LOCK_AGE_INT=%%I
    if !LOCK_AGE_INT! GEQ 2 (
        echo [%date% %time%] Clearing stale lock file ^(!LOCK_AGE_INT!h old^) >> "%LOGFILE%"
        del "%LOCKFILE%" 2>nul
    ) else (
        echo [%date% %time%] Skipping: previous run still active ^(lock age: !LOCK_AGE_INT!h^) >> "%LOGFILE%"
        exit /b 0
    )
)

REM Create lock file
echo %date% %time% > "%LOCKFILE%"

REM Run disappearance tracker with large batch
echo [%date% %time%] Starting disappearance check >> "%LOGFILE%"
cd /d "%SRCDIR%"
"%PYTHON%" disappearance_tracker.py --recheck --min-age 4h --max-ids 2000 >> "%LOGFILE%" 2>&1
echo [%date% %time%] Disappearance check complete (exit code: %ERRORLEVEL%) >> "%LOGFILE%"

REM Remove lock file
del "%LOCKFILE%" 2>nul

endlocal
