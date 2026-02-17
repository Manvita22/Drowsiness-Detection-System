@echo off
REM Drowsiness Detection System - Setup Script for Windows

echo.
echo ========================================
echo Drowsiness Detection System Setup
echo ========================================
echo.

REM Check Python installation
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.8+ from https://www.python.org/
    pause
    exit /b 1
)

echo [1/4] Python found: 
python --version

REM Install requirements
echo.
echo [2/4] Installing required packages...
pip install -r requirements.txt
if errorlevel 1 (
    echo Error: Failed to install packages
    pause
    exit /b 1
)

REM Create models directory
echo.
echo [3/4] Creating directories...
if not exist "models" mkdir models
echo Models directory created

REM Display next steps
echo.
echo [4/4] Setup Complete!
echo.
echo ========================================
echo Next Steps:
echo ========================================
echo.
echo Option 1: Train the model (first time)
echo   python train_model.py
echo.
echo Option 2: Test on test dataset
echo   python test_model.py
echo.
echo Option 3: Run detection system (requires webcam)
echo   python drowsiness_detection.py
echo.
echo Notes:
echo - Training takes 5-10 minutes on CPU
echo - For faster training, use a GPU system
echo - Place alarm.wav in project root for audio alerts
echo - Press 'q' to exit detection system
echo.
pause
