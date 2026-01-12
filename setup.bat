@echo off
setlocal enabledelayedexpansion

echo.
echo ======================================================
echo   FableYard Studio - Setup
echo ======================================================
echo.

REM Check Python 3.11+
echo [1/5] Checking Python version...
python --version 2>nul | findstr /R "Python 3\.1[1-9]" >nul
if errorlevel 1 (
    echo [ERROR] Python 3.11+ required
    echo Please install Python 3.11 or later from https://www.python.org/
    pause
    exit /b 1
)
echo [OK] Python version check passed
echo.

REM Create shared venv at project root
echo [2/5] Creating virtual environment...
if not exist .venv (
    python -m venv .venv
    if errorlevel 1 (
        echo [ERROR] Failed to create virtual environment
        pause
        exit /b 1
    )
    echo [OK] Virtual environment created
) else (
    echo [OK] Virtual environment already exists
)
echo.

REM Activate venv
echo [3/5] Installing Python dependencies...
call .venv\Scripts\activate.bat
if errorlevel 1 (
    echo [ERROR] Failed to activate virtual environment
    pause
    exit /b 1
)

REM Install Python dependencies
pip install --upgrade pip >nul 2>&1
pip install -r requirements.txt
if errorlevel 1 (
    echo [ERROR] Failed to install Python dependencies
    pause
    exit /b 1
)
echo [OK] Python dependencies installed
echo.

REM Create user directory structure
echo [4/5] Creating user directory structure...
if not exist user mkdir user
if not exist user\adapters mkdir user\adapters
if not exist user\adapters\flux mkdir user\adapters\flux
if not exist user\adapters\z mkdir user\adapters\z
if not exist user\models mkdir user\models
if not exist user\models\txt2img mkdir user\models\txt2img
if not exist user\models\txt2img\flux mkdir user\models\txt2img\flux
if not exist user\models\txt2img\z mkdir user\models\txt2img\z
if not exist user\outputs mkdir user\outputs
echo [OK] User directories created
echo.

REM Install UI dependencies
echo [5/5] Installing UI dependencies...
cd ui
if not exist package.json (
    echo [ERROR] UI package.json not found
    pause
    exit /b 1
)

call npm install
if errorlevel 1 (
    echo [ERROR] Failed to install UI dependencies
    pause
    exit /b 1
)
cd ..
echo [OK] UI dependencies installed
echo.

echo ======================================================
echo   Setup Complete!
echo ======================================================
echo.
echo Next steps:
echo   1. Place your models in: user/models/
echo   2. Run: start.bat
echo.
pause
