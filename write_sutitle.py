import os
import subprocess

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
                lines = f.readlines()
            
            chinese_lines = []
            i = 0
            while i < len(lines):
                line = lines[i].strip()
                
                # Skip empty lines
                if not line:
                    i += 1
                    continue
                
                # Check if this is a subtitle number
                if line.isdigit():
                    chinese_lines.append(line + '\n')  # Subtitle number
                    i += 1
                    
                    # Get timestamp line
                    if i < len(lines):
                        chinese_lines.append(lines[i])  # Timestamp
                        i += 1
                    
                    # Get Chinese text (first text line)
                    if i < len(lines):
                        chinese_text = lines[i].strip()
                        if chinese_text:
                            chinese_lines.append(chinese_text + '\n')
                        i += 1
                    
                    # Skip Japanese text (second text line)
                    if i < len(lines):
                        i += 1
                    
                    # Add empty line
                    chinese_lines.append('\n')
                else:
                    i += 1
            
            # Write Chinese-only SRT
            with open(chinese_only_srt_path, 'w', encoding='utf-8') as f:
                f.writelines(chinese_lines)
            
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
            
            ffmpeg_bin = 'C:/ffmpeg/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe' #请把这里的地址改成你自己电脑里ffmpeg.exe的地址
            # Using POSIX style path
            srt_escaped = self._escape_path_for_ffmpeg_subtitles(srt_to_use)

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

