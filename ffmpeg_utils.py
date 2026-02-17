"""Shared FFmpeg / FFprobe helpers used by multiple modules."""

import os
import shutil
import subprocess


def subprocess_kwargs():
    """Return extra kwargs for subprocess calls (hides console window on Windows)."""
    kwargs = {}
    if os.name == "nt":
        kwargs["creationflags"] = subprocess.CREATE_NO_WINDOW
    return kwargs


def find_ffmpeg():
    """Locate the ffmpeg binary (PATH first, then Windows fallback)."""
    ffmpeg_bin = shutil.which("ffmpeg")
    if ffmpeg_bin:
        return ffmpeg_bin
    fallback = "C:/ffmpeg/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe"
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError("ffmpeg not found in PATH or fallback path")


def find_ffprobe():
    """Locate the ffprobe binary (PATH first, then Windows fallback)."""
    ffprobe_bin = shutil.which("ffprobe")
    if ffprobe_bin:
        return ffprobe_bin
    fallback = "C:/ffmpeg/ffmpeg-master-latest-win64-gpl/bin/ffprobe.exe"
    if os.path.exists(fallback):
        return fallback
    raise FileNotFoundError("ffprobe not found in PATH or fallback path")
