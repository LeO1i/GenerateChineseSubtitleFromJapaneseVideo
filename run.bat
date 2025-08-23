@echo off
echo Japanese Video Subtitle Generator
echo =================================
echo.
echo Starting GUI...
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python is not installed or not in PATH
    echo Please install Python 3.9+ from https://python.org
    pause
    exit /b 1
)

REM Try to activate virtual environment if it exists
if exist ".venv\Scripts\Activate.bat" (
    echo Activating virtual environment...
    call .venv\Scripts\Activate.bat
    echo Virtual environment activated.
) else if exist "venv\Scripts\Activate.bat" (
    echo Activating virtual environment...
    call venv\Scripts\Activate.bat
    echo Virtual environment activated.
) else (
    echo No virtual environment found, using system Python.
)

REM Run the GUI
python gui.py

REM If GUI fails, show error
if errorlevel 1 (
    echo.
    echo Error: GUI failed to start.
    echo Please check that all dependencies are installed:
    echo pip install -r requirements.txt
    echo.
    pause
    exit /b 1
)

echo.
echo Press any key to exit...
pause >nul
