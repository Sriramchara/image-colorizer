@echo off
cd /d "%~dp0"
echo Starting Image Colorizer with GPU Support...
py -3.10 src/app.py
pause
