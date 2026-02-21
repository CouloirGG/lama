@echo off
title LAMA - Release Changelog
cd /d "%~dp0.."
echo.
echo ================================================
echo   LAMA — Release Changelog Generator
echo ================================================
echo.
python scripts\release.py %*
echo.
pause
