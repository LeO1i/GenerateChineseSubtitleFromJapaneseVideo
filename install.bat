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

REM Install PyTorch (prefer CUDA build; fallback to CPU if needed)
echo.
echo Installing PyTorch CUDA build (cu128)...
pip uninstall -y torch torchvision torchaudio >nul 2>&1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo CUDA PyTorch installation failed, falling back to default PyTorch...
    pip install torch
    if errorlevel 1 (
        echo PyTorch installation failed
        pause
        exit /b 1
    )
    echo Warning: Installed fallback PyTorch package (may be CPU-only).
)

echo PyTorch installed ✓

REM Install other dependencies
echo.
echo Installing Hugging Face dependencies...
pip install -r requirements.txt
if errorlevel 1 (
    echo Dependency installation failed
    pause
    exit /b 1
)

echo Dependencies installed ✓

echo.
echo Verifying key imports...
python -c "import torch, transformers, accelerate, tokenizers, safetensors, sentencepiece, qwen_asr" >nul 2>&1
if errorlevel 1 (
    echo Import check failed. Please run: pip install -r requirements.txt
    pause
    exit /b 1
)
echo Import check passed ✓
python -c "import torch; print('Torch version:', torch.__version__); print('CUDA available:', torch.cuda.is_available()); print('Torch CUDA:', torch.version.cuda)"

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
