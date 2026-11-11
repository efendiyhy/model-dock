@echo off
cd /d "%~dp0"
echo [ModelDock] Setting up local environment...
if not exist ".venv\Scripts\python.exe" (
    py -3 -m venv .venv
    if errorlevel 1 (python -m venv .venv)
)
if not exist ".venv\Scripts\python.exe" (
    echo Python 3.10+ is required. Install Python from python.org and try again.
    pause
    exit /b 1
)
".venv\Scripts\python.exe" -m pip install -r requirements.txt
if errorlevel 1 (echo Dependency installation failed. & pause & exit /b 1)
echo.
echo [ModelDock] Open in browser: http://127.0.0.1:8000
start "" "http://127.0.0.1:8000"
".venv\Scripts\python.exe" -m uvicorn app.main:app --host 127.0.0.1 --port 8000
pause
