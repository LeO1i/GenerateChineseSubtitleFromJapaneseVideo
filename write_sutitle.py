import os
import subprocess

from ffmpeg_utils import find_ffmpeg, subprocess_kwargs


class WriteSubtitle:
    def __init__(self):
        pass

    def _escape_path_for_ffmpeg_subtitles(self, path):
        """
        Return a POSIX-style absolute path for FFmpeg subtitles filter with escaped specials.

        - Use forward slashes to avoid backslash escaping
        - Escape the Windows drive colon (e.g. D\\:/path/file.srt)
        - Escape single quotes if any
        """
        from pathlib import Path
        posix_path = Path(path).resolve().as_posix()
        posix_path = posix_path.replace(':', '\\:')
        posix_path = posix_path.replace("'", "\\'")
        return posix_path

    def extract_chinese_from_bilingual_srt(self, bilingual_srt_path, chinese_only_srt_path):
        """
        Extract Chinese-only subtitles from a bilingual SRT file
        """
        print("Extracting Chinese subtitles from bilingual SRT...")
        try:
            with open(bilingual_srt_path, 'r', encoding='utf-8') as f:
                blocks = f.read().split('\n\n')

            chinese_blocks = []
            for block in blocks:
                lines = [line for line in block.splitlines() if line.strip()]
                if len(lines) < 2:
                    continue

                # Standard SRT block:
                # 0: index
                # 1: timestamp
                # 2+: text lines
                timestamp_line = lines[1]
                text_lines = lines[2:]
                if not text_lines:
                    continue

                # Generated bilingual format is JA first, ZH second.
                # Keep second line when present; otherwise fallback to first line.
                chinese_text = text_lines[1].strip() if len(text_lines) >= 2 else text_lines[0].strip()
                if not chinese_text:
                    continue

                chinese_blocks.append((timestamp_line, chinese_text))
            
            # Write Chinese-only SRT
            with open(chinese_only_srt_path, 'w', encoding='utf-8') as f:
                for idx, (timestamp_line, chinese_text) in enumerate(chinese_blocks, start=1):
                    f.write(f"{idx}\n")
                    f.write(f"{timestamp_line}\n")
                    f.write(f"{chinese_text}\n\n")
            
            print(f"Chinese-only SRT extracted: {chinese_only_srt_path}")
            return chinese_only_srt_path
            
        except Exception as e:
            print(f"Error extracting Chinese subtitles: {e}")
            return None

    def burn_subtitles(self, video_path, srt_path, output_path, font_name='Microsoft YaHei'):
        """
        Use FFmpeg to burn the chinese subtitle to the bottom of video and output the new video
        """
        print("Burning subtitle to video...")
        try:
            # Check if this is a bilingual SRT file and extract Chinese-only if needed
            temp_chinese_srt = None
            srt_to_use = srt_path
            
            # Read first few lines to check if it's bilingual
            with open(srt_path, 'r', encoding='utf-8') as f:
                lines = f.readlines()
            
            # Check if it's bilingual (has both Chinese and Japanese)
            is_bilingual = False
            for i, line in enumerate(lines):
                if i > 20:  # Check first 20 lines
                    break
                line = line.strip()
                # Look for Japanese characters
                if any(ord(char) > 127 for char in line) and any(0x3040 <= ord(char) <= 0x309F or 0x30A0 <= ord(char) <= 0x30FF for char in line):
                    is_bilingual = True
                    break
            
            if is_bilingual:
                print("Detected bilingual SRT file, extracting Chinese subtitles...")
                temp_chinese_srt = srt_path.replace('.srt', '_temp_chinese.srt')
                extracted_srt = self.extract_chinese_from_bilingual_srt(srt_path, temp_chinese_srt)
                if extracted_srt:
                    srt_to_use = extracted_srt
                else:
                    print("Warning: Failed to extract Chinese subtitles, using original file")
            
            ffmpeg_bin = find_ffmpeg()
            # Using POSIX style path
            srt_escaped = self._escape_path_for_ffmpeg_subtitles(srt_to_use)

            safe_font = font_name.replace("'", "\'")
            # 15% smaller than previous 28.
            font_size = 24
            # Lock subtitles to the lower half and near the bottom.
            # Use a conservative fixed margin so libass keeps bottom placement.
            margin_v = 40
            force_style = (
                f"'FontName={safe_font},FontSize={font_size},Alignment=2,BorderStyle=1,"
                f"PrimaryColour=&H00FFFFFF,OutlineColour=&H00000000,"
                f"Outline=2,Shadow=0,MarginV={margin_v}'"
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
            subprocess.run(cmd, check=True, capture_output=True, **subprocess_kwargs())
            print(f"Burning finished: {output_path}")
            
            # Clean up temporary file
            if temp_chinese_srt and os.path.exists(temp_chinese_srt):
                os.remove(temp_chinese_srt)
            
            return True
        except subprocess.CalledProcessError as e:
            try:
                stderr = e.stderr.decode('utf-8', errors='ignore') if e.stderr else ''
            except Exception:
                stderr = str(e)
            print(f"Burning fail: {e}\nFFmpeg Wrong Output:\n{stderr}")
            
            # Clean up temporary file on error
            if temp_chinese_srt and os.path.exists(temp_chinese_srt):
                os.remove(temp_chinese_srt)
            
            return False

