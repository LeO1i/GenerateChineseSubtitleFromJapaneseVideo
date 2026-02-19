@echo off
chcp 65001 >nul
echo 日语字幕生成器
echo ==============
echo.

REM Check if Python is available
python --version >nul 2>&1
if errorlevel 1 (
    echo 错误：未找到 Python
    echo 请先运行 install.bat 安装依赖
    echo.
    pause
    exit /b 1
)

REM Check if required packages are installed
python -c "import torch, transformers, accelerate, tokenizers, safetensors, sentencepiece, qwen_asr" >nul 2>&1
if errorlevel 1 (
    echo 缺少必要依赖，正在安装...
    pip install -r requirements.txt
    if errorlevel 1 (
        echo 依赖安装失败
        echo 请运行：pip install -r requirements.txt
        pause
        exit /b 1
    )
)

REM Check if FFmpeg is available
ffmpeg -version >nul 2>&1
if errorlevel 1 (
    echo 警告：未找到 FFmpeg
    echo 请从 https://ffmpeg.org/download.html 下载并安装 FFmpeg
)

echo 正在启动程序...

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
