@echo off
cd /d %~dp0
echo Current Directory: %CD%

if not exist train_classifier.py (
    echo ERROR: train_classifier.py not found in %CD%
    pause
    exit /b 1
)

set PYTHON_CMD=
echo Checking for 'python'...
python --version >nul 2>&1
if %errorlevel% equ 0 set PYTHON_CMD=python

if not defined PYTHON_CMD (
    echo Checking for 'py'...
    py --version >nul 2>&1
    if %errorlevel% equ 0 set PYTHON_CMD=py
)

if not defined PYTHON_CMD (
    echo Checking for 'python3'...
    python3 --version >nul 2>&1
    if %errorlevel% equ 0 set PYTHON_CMD=python3
)

if not defined PYTHON_CMD (
    echo ERROR: No Python interpreter found (checked python, py, python3).
    echo Attempting to run with 'python' anyway to show error...
    python train_classifier.py --data_dir dataset/food_classification --epochs 10
) else (
    echo Found Python: %PYTHON_CMD%
    %PYTHON_CMD% train_classifier.py --data_dir dataset/food_classification --epochs 10
)

if %errorlevel% neq 0 (
    echo.
    echo ERROR: Script failed with exit code %errorlevel%.
) else (
    echo.
    echo Training complete successfully.
)
pause
