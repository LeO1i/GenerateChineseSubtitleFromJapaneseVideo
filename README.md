# Japanese Subtitle Generator (Windows)

This is a Japanese video subtitle generator for Windows. It extracts audio from Japanese videos, generates Japanese subtitles using Whisper, and translates them into Chinese to produce a bilingual SRT.

## System Requirements

- Windows 10/11
- Python 3.9 or later
- FFmpeg (for video processing)

## Installation

1. Install Python 3.9+
   - Download from https://python.org and install
   - Check "Add Python to PATH" during installation

2. Install FFmpeg
   - Download Windows build from https://ffmpeg.org/download.html
   - Extract to `C:\\ffmpeg\\`
   - Add `C:\\ffmpeg\\bin` to the system PATH

3. Install dependencies
   - Double-click `install.bat` to auto-install
   - Or run: `pip install -r requirements.txt`

## Usage

1. Double-click `run.bat` to start the GUI
2. Select a Japanese video file to process
3. Choose a Whisper model (medium recommended)
4. Click "Generate Subtitles"
5. After processing, a bilingual SRT (JA + ZH) will be created

## Files

- `gui.py` - GUI application
- `main.py` - Command-line version
- `speech_extract.py` - Speech recognition and subtitle generation
- `write_sutitle.py` - Burn subtitles to video
- `llmtranslate.py` - LLM translation example
- `install.bat` - Automatic installer
- `run.bat` - Start script

## Notes

- First run downloads the Whisper model; internet required
- Processing time depends on video length and selected model
- Clear audio improves recognition quality
- You can edit the generated SRT before burning it into the video

## Troubleshooting

If you encounter problems:
1. Ensure Python and FFmpeg are correctly installed
2. Check your internet connection (first run downloads the model)
3. Try re-running `install.bat`
4. Check if the video file format is supported

## Supported Formats

- Video formats: MP4, MKV, AVI, MOV, WMV, FLV
- Output formats: SRT subtitle files, MP4 video files (with hardsubs)
