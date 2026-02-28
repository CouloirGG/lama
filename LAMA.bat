@echo off
if exist "%~dp0dist\LAMA\LAMA.exe" (
    start "" "%~dp0dist\LAMA\LAMA.exe"
) else (
    start "" pythonw "%~dp0src\app.py"
)
