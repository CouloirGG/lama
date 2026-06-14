@echo off
echo === LAMA Debug Mode ===
echo DevTools: right-click anywhere in the window and select "Inspect"
echo Console output will appear in this terminal.
echo.
python "%~dp0src\app.py" --debug
pause
