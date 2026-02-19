@echo off
chcp 65001 >nul
echo 日语字幕生成器 - 安装脚本
echo ==========================
echo.

REM Check if Python is available
echo 正在检查 Python 安装情况...
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到 Python
    echo.
    echo 请安装 Python 3.9 或更高版本：
    echo 1. 访问 https://python.org
    echo 2. 下载并安装 Python 3.9+
    echo 3. 安装时勾选 "Add Python to PATH"
    echo 4. 安装完成后重新运行本脚本
    echo.
    pause
    exit /b 1
)

echo 已找到 Python ✓

REM Install PyTorch (prefer CUDA build; fallback to CPU if needed)
echo.
echo 正在安装 PyTorch CUDA 版本（cu128）...
pip uninstall -y torch torchvision torchaudio >nul 2>&1
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu128
if errorlevel 1 (
    echo CUDA 版 PyTorch 安装失败，正在回退到默认 PyTorch...
    pip install torch
    if errorlevel 1 (
        echo PyTorch 安装失败
        pause
        exit /b 1
    )
    echo 警告：已安装回退版 PyTorch（可能为 CPU-only）。
)

echo PyTorch 安装完成 ✓

REM Install other dependencies
echo.
echo 正在安装 Hugging Face 相关依赖...
pip install -r requirements.txt
if errorlevel 1 (
    echo 依赖安装失败
    pause
    exit /b 1
)

echo 依赖安装完成 ✓

echo.
echo 正在验证关键依赖导入...
python -c "import torch, transformers, accelerate, tokenizers, safetensors, sentencepiece, qwen_asr" >nul 2>&1
if errorlevel 1 (
    echo 导入检查失败。请运行：pip install -r requirements.txt
    pause
    exit /b 1
)
echo 导入检查通过 ✓
python -c "import torch; print('Torch 版本：', torch.__version__); print('CUDA 是否可用：', torch.cuda.is_available()); print('Torch CUDA：', torch.version.cuda)"

REM Check FFmpeg
echo.
echo 正在检查 FFmpeg...
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo 警告：未找到 FFmpeg
    echo.
    echo 请安装 FFmpeg：
    echo 1. 访问 https://ffmpeg.org/download.html
    echo 2. 下载 Windows 版本
    echo 3. 解压到 C:\ffmpeg\
    echo 4. 将 C:\ffmpeg\bin 加入系统 PATH
) else (
    echo 已找到 FFmpeg ✓
)

echo.
echo 安装完成！
echo.
echo 现在可以双击 run.bat 启动程序
echo.
pause
