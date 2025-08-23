## 日语视频字幕生成器

使用 OpenAI Whisper 从日语视频生成中文字幕，可选择查看/编辑 SRT 文件，并使用 FFmpeg 将字幕烧录回视频中。 
### 推荐: 0编程基础的使用者在AI的帮忙下完成安装, 否则可能需要一些时间完成安装.


### 功能特点
- **全新：现代图形界面** - 易于使用的图形界面，支持文件浏览和进度跟踪
- **全新：模型选择** - 选择不同的 Whisper 模型（tiny 到 large-v3）以平衡准确性和速度
- **全新：GPU/CPU 状态显示** - 显示您使用的是 GPU 还是 CPU 进行处理
- **全新：智能 SRT 输出** - 生成一个双语 SRT 文件，自动提取中文用于烧录
- **全新：一键启动** - 无需菜单选择，直接访问图形界面
- 使用 Whisper 进行自动语音识别（日语 → 文本）
- 通过 Google Translate 进行自动翻译（日语 → 中文）
- 具有更好的换行和时序的 SRT 生成
- 两步工作流程：
  1) 生成 SRT 并查看/编辑
  2) 将 SRT 烧录到视频中（硬字幕）
- 一键传统流程在代码中仍然可用（如需要）

### 系统要求
- Python 3.9+（推荐 64 位）
- 本地安装 FFmpeg 并可在已知路径访问
- 互联网连接用于 Google Translate
- tkinter（通常随 Python 一起安装）

### 安装步骤
1) 创建虚拟环境（Windows PowerShell）：
```powershell
python -m venv .venv
.\.venv\Scripts\Activate.ps1
```

2) 安装 PyTorch
- CPU（简单）：
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cpu
```
- GPU（CUDA）：按照 [PyTorch 入门](https://pytorch.org/get-started/locally/) 的官方选择器，例如（相应替换 cu121）：
```powershell
pip install torch --index-url https://download.pytorch.org/whl/cu121
```

3) 安装项目依赖：
```powershell
pip install -r requirements.txt
```

4) 安装 FFmpeg（Windows）
- 从 [FFmpeg 官方网站](https://ffmpeg.org/download.html) 或可信镜像下载静态构建
- 解压到：`C:\ffmpeg\ffmpeg-master-latest-win64-gpl\`（匹配代码中的默认路径），或
- 将 `ffmpeg.exe` 放在您的 PATH 中，然后更新代码中的硬编码路径为 `ffmpeg`

如果您的 FFmpeg 路径不同，请更新这两个地方：
- `write_sutitle.py`：变量 `ffmpeg_bin`
- `speech_extract.py`：在 `extract_audio(...)` 命令列表中

### 使用方法

#### 图形界面模式（推荐）
**一键启动（Windows）：**
```powershell
run.bat
```

**一键启动（Linux/Mac）：**
```bash
./run.sh
```

**手动启动：**
```powershell
python gui.py
```

**替代方案（命令行模式）：**
```powershell
python main.py
```

**图形界面功能：**
- 浏览和选择视频文件（MP4、MKV、AVI、MOV、WMV、FLV）
- 浏览和选择现有 SRT 文件进行烧录
- 选择生成文件的输出目录
- 选择 Whisper 模型（tiny、base、small、medium、large、large-v2、large-v3）
- **GPU/CPU 状态显示** - 显示处理设备和性能说明
- **智能 SRT 输出** - 生成一个双语文件，自动提取中文用于烧录
- 实时进度跟踪和日志记录
- 线程处理（处理期间图形界面保持响应）

#### 命令行模式
运行交互式程序：
```powershell
python main.py
```

系统会询问：
- 您是否已经有此程序生成的 SRT
  - 如果是：提供 SRT 路径和视频路径直接烧录
  - 如果否：仅提供视频路径；应用程序将首先生成 SRT

典型流程（推荐）：
1) 当询问是否已有 SRT 时选择 "n"
2) 提供日语视频路径
3) 等待转录和翻译；双语 SRT 将保存在视频旁边（例如 `yourvideo_chinese.srt`）
4) 在任何字幕编辑器或文本编辑器中打开并查看/编辑 SRT
5) 当提示时，选择 "y" 将 SRT 烧录到新视频中（仅烧录中文文本）

### Whisper 模型选择
- **tiny**：最快，准确性最低（39M 参数）
- **base**：快速，准确性低（74M 参数）
- **small**：平衡速度和准确性（244M 参数）
- **medium**：良好的准确性，中等速度（769M 参数）- **默认**
- **large**：高准确性，较慢（1550M 参数）
- **large-v2**：很高准确性，慢（1550M 参数）
- **large-v3**：最高准确性，最慢（1550M 参数）

### 注意事项
- 应用程序自动检测 CUDA。如果没有 CUDA 工具包/驱动程序，将在 CPU 上运行（较慢）。要启用 GPU，请安装支持 CUDA 的 PyTorch 构建。
- 确保您的 SRT 保存为 UTF-8。FFmpeg 过滤器配置为 `charenc=UTF-8`。
- 字幕样式可以在 `write_sutitle.py` 中通过 `force_style`（字体、大小、轮廓、边距）进行调整。
- 图形界面处理在后台线程中运行，保持界面响应。

### 项目结构
- `gui.py`：现代图形界面，支持文件浏览和模型选择（主界面）
- `main.py`：命令行入口。指导两步工作流程（生成 → 查看 → 烧录）
- `speech_extract.py`：核心管道（音频提取、Whisper 转录、翻译、SRT 写入）
- `write_sutitle.py`：使用 FFmpeg 将 SRT 烧录到视频中
- `run.bat` / `run.sh`：带虚拟环境激活的一键启动器

### 故障排除
- **使用 run.bat 时未检测到 GPU：**
  - 批处理文件现在自动激活虚拟环境
  - 如果仍然不工作，运行 `python test_gpu.py` 检查 GPU 检测
  - 确保在虚拟环境中安装了支持 CUDA 的 PyTorch
- **找不到 FFmpeg / 路径错误：**
  - 验证 `write_sutitle.py` 中的 `ffmpeg_bin` 路径和 `speech_extract.py` → `extract_audio(...)` 中的路径
  - 尝试在 shell 中运行 `ffmpeg -version` 确认它在 PATH 中
- **性能缓慢：**
  - 在 CPU 上运行要慢得多。如果您有兼容的 NVIDIA GPU，请安装支持 CUDA 的 PyTorch
  - 尝试较小的 Whisper 模型（tiny、base、small）以获得更快的处理速度
- **Google Translate 间歇性失败：**
  - `googletrans` 库使用非官方 API 可能会限制。您可以重新运行或在需要时添加自己的翻译后端
- **图形界面不工作：**
  - 确保安装了 tkinter：`python -c "import tkinter; print('tkinter available')"`
  - 在 Linux 上，如果缺少则安装 python-tk 包
  - 在 Windows 上，使用来自 python.org 的 Python，其中包含 tkinter

### 许可证
仅供个人/教育使用。在商业使用前请检查第三方工具（Whisper、PyTorch、FFmpeg、Google Translate）的许可证。


