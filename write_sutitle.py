import os
import subprocess

class WriteSubtitle:
    def __init__(self):
        pass
    
    def _escape_path_for_ffmpeg_subtitles(self, path):
        """
        Return a POSIX-style absolute path for FFmpeg subtitles filter with escaped specials.

        - Use forward slashes to avoid backslash escaping
        - Escape the Windows drive colon (e.g. D\:/path/file.srt)
        - Escape single quotes if any
        """
        from pathlib import Path
        posix_path = Path(path).resolve().as_posix()
        posix_path = posix_path.replace(':', '\\:')
        posix_path = posix_path.replace("'", "\\'")
        return posix_path

    def burn_subtitles(self, video_path, srt_path, output_path, font_name='Microsoft YaHei'):
        """
        Use FFmpeg to burn the chinese subtitle to the bottom of video and output the new video
        """
        print("Burning subtitle to video...")
        try:
            ffmpeg_bin = 'C:/ffmpeg/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe'
            # Using POSIX style path
            srt_escaped = self._escape_path_for_ffmpeg_subtitles(srt_path)

            safe_font = font_name.replace("'", "\'")
            force_style = (
                f"'FontName={safe_font},FontSize=28,BorderStyle=3,Outline=2,Shadow=0,OutlineColour=&H80000000&,MarginV=36'"
            )

            vf_arg = f"subtitles=filename='{srt_escaped}':charenc=UTF-8:force_style={force_style}"

            cmd = [
                ffmpeg_bin,
                '-hide_banner',
                '-i', os.path.abspath(video_path),
                '-vf', vf_arg,
                '-c:v', 'libx264',
                '-crf', '20',
                '-preset', 'medium',
                '-c:a', 'copy',
                output_path,
                '-y'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Burning finished: {output_path}")
            return True
        except subprocess.CalledProcessError as e:
            try:
                stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
            except Exception:
                stderr = str(e)
            print(f"Burning fail: {e}\nFFmpeg Wrong Output:\n{stderr}")
            return False

