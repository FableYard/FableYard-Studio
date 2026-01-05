@echo off
setlocal enabledelayedexpansion

echo.
echo ======================================================
echo   FableYard Studio
echo ======================================================
echo.

REM Check if core venv exists (has CUDA support)
if not exist core\.venv (
    echo [ERROR] Core virtual environment not found
    echo Please ensure core/.venv exists with CUDA PyTorch
    pause
    exit /b 1
)

REM Start API + Worker using core venv (has CUDA)
echo [1/2] Starting API and Worker...
start "FableYard API+Worker" core\.venv\Scripts\python.exe start.py

REM Wait for API to be ready
echo [1/2] Waiting for API to start...
timeout /t 5 /nobreak >nul

REM Start UI in new window
echo [2/2] Starting UI...
cd ui
start "FableYard UI" npm run dev
cd ..

echo.
echo ======================================================
echo   FableYard Studio is starting!
echo ======================================================
echo.
echo   API:  http://localhost:8000
echo   UI:   http://localhost:5173 (or similar)
echo.
echo Press any key to stop all services...
pause >nul

REM Kill all services
echo.
echo Stopping services...
taskkill /FI "WINDOWTITLE eq FableYard*" /F >nul 2>&1

echo.
echo All services stopped.
echo.
