@echo off
chcp 65001 >nul
echo Japanese Subtitle Generator - Installer
echo =============================
echo.

REM Check if Python is available
echo Checking Python installation...
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    echo.
    echo Please install Python 3.9 or later:
    echo 1. Visit https://python.org
    echo 2. Download and install Python 3.9+
    echo 3. Check "Add Python to PATH" during install
    echo 4. Re-run this installer
    echo.
    pause
    exit /b 1
)

echo Python found ✓

REM Install PyTorch (CPU version for Windows)
echo.
echo Installing PyTorch (CPU build)...
pip install torch --index-url https://download.pytorch.org/whl/cpu
if errorlevel 1 (
    echo PyTorch installation failed
    pause
    exit /b 1
)

echo PyTorch installed ✓

REM Install other dependencies
echo.
echo Installing other dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed
    pause
    exit /b 1
)

echo Dependencies installed ✓

REM Check FFmpeg
echo.
echo Checking FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo Warning: FFmpeg not found
    echo.
    echo Please install FFmpeg:
    echo 1. Visit https://ffmpeg.org/download.html
    echo 2. Download Windows build
    echo 3. Extract to C:\ffmpeg\
    echo 4. Add C:\ffmpeg\bin to system PATH
) else (
    echo FFmpeg found ✓
)

echo.
echo Installation completed!
echo.
echo You can now double-click run.bat to start the program
echo.
pause
