@echo off
chcp 65001 >nul
echo Japanese Subtitle Generator
echo ===================
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo Error: Python not found
    echo Please run install.bat to install dependencies
    echo.
    pause
    exit /b 1
)

REM Check if required packages are installed
python -c "import torch, transformers, accelerate, tokenizers, safetensors, sentencepiece, qwen_asr" >nul 2>&1
if errorlevel 1 (
    echo Missing required dependencies, installing...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo Dependency installation failed
        echo Please run: pip install -r requirements.txt
        pause
        exit /b 1
    )
)

REM Check if FFmpeg is available
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo Warning: FFmpeg not found
    echo Please download and install FFmpeg from https://ffmpeg.org/download.html
)

echo Starting application...

REM Launch GUI detached so this window can close without affecting the app
REM Prefer pythonw (no console). Fallback to python if pythonw not available.
where pythonw >nul 2>&1
if %errorlevel%==0 (
    start "" pythonw gui.py
) else (
    start "" python gui.py
)

REM Exit the launcher window now
exit /b 0
