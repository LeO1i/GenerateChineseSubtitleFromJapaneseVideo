import os
import subprocess
import whisper
from googletrans import Translator
import datetime
import torch

device = "cuda:0" if torch.cuda.is_available() else "cpu"


class JapaneseVideoSubtitleGenerator:
    def __init__(self):
        # Initialize Whisper model (for speech recognition)
        print("Loading Whisper model...")
        self.whisper_model = whisper.load_model("medium", device=device)
        
        # Initialize Google translator
        self.translator = Translator()
        
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

    def extract_audio(self, video_path, audio_path):
        """
        Enhanced audio extraction with noise reduction
        """
        print("Extracting audio from video...")
        try:
            cmd = [
                'C:/ffmpeg/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe',
                '-i', video_path,
                # Enhanced audio processing
                '-acodec', 'pcm_s16le',
                '-ar', '16000',  # Whisper's preferred sample rate
                '-ac', '1',      # Mono audio
                # Audio filters for better quality
                '-af', 'highpass=f=80,lowpass=f=8000,volume=1.5,dynaudnorm',
                audio_path, '-y'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Audio extraction completed: {audio_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Audio extraction failed: {e}")
            return False
    
    def transcribe_japanese(self, audio_path):
        """
        Enhanced Japanese speech recognition with better parameters
        """
        print("Starting Japanese speech recognition...")
        try:
            # Use larger model for better accuracy
            result = self.whisper_model.transcribe(
                audio_path,
                language='ja',
                task='transcribe',
                # Enhanced parameters for better accuracy
                word_timestamps=True,  # Get word-level timestamps
                temperature=0.0,       # Deterministic output
                beam_size=5,          # Better beam search
                best_of=5,            # Multiple attempts
                patience=1.0,         # Wait for better results
                condition_on_previous_text=True,  # Use context
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            print("Japanese speech recognition completed")
            return result
        except Exception as e:
            print(f"Speech recognition failed: {e}")
            return None
    
    def translate_to_chinese(self, japanese_text):
        """
        Translate Japanese to Chinese
        """
        try:
            cleaned_text = self.clean_japanese_text(japanese_text)
            result = self.translator.translate(cleaned_text, src='ja', dest='zh-cn')
            translated = result.text
            
            # Post-process the translation
            translated = self.post_process_translation(translated, japanese_text)
            
            return translated
        except Exception as e:
            print(f"Translation failed: {e}")
            return japanese_text  # If translation fails, return the original text

    def clean_japanese_text(self, text):
        """
        Clean Japanese text for better translation
        """
        import re
        # Remove extra whitespace
        text = re.sub(r'\s+', ' ', text.strip())
        # Remove speaker indicators like "A:" "B:"
        text = re.sub(r'^[A-Z]:\s*', '', text)
        # Remove common filler sounds
        text = re.sub(r'[ええ]{2,}|[あああ]{2,}|[うううう]{2,}', '', text)
        return text

    def post_process_translation(self, translated, original):
        """
        Post-process translation for better quality
        """
        # Fix common translation issues
        translated = translated.replace('。。', '。')
        translated = translated.replace('，，', '，')
        
        # If translation is too short compared to original, it might be incomplete
        if len(translated) < len(original) * 0.3 and len(original) > 10:
            print(f"Warning: Translation may be incomplete - Original: {original[:50]}...")
        
        return translated            
    
    def format_time(self, seconds):
        """
        Convert seconds to SRT format time
        """
        td = datetime.timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        seconds = td.total_seconds() % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"
    
    def process_segments_with_validation(self, transcription_result):
        """
        Process segments with validation and error correction
        """
        processed_segments = []
        
        for segment in transcription_result['segments']:
            # Skip very short segments (likely noise)
            if segment['end'] - segment['start'] < 0.5:
                continue
                
            # Skip segments with very low confidence
            if hasattr(segment, 'confidence') and segment.get('confidence', 1.0) < 0.3:
                continue
                
            # Merge very short adjacent segments
            if (processed_segments and 
                segment['start'] - processed_segments[-1]['end'] < 1.0 and
                len(segment['text'].strip()) < 10):
                # Merge with previous segment
                processed_segments[-1]['text'] += ' ' + segment['text']
                processed_segments[-1]['end'] = segment['end']
            else:
                processed_segments.append(segment)
        
        return processed_segments 

    def generate_srt(self, transcription_result, output_path):
        """
        Generate SRT with enhanced formatting and validation
        """
        print("Generating Chinese subtitles...")
        
        # Process segments first
        segments = self.process_segments_with_validation(transcription_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments):
                japanese_text = segment['text'].strip()
                
                if not japanese_text:
                    continue
                
                # Enhanced translation
                chinese_text = self.translate_to_chinese(japanese_text)
                
                # Format time with validation
                start_time = self.format_time(max(0, segment['start']))
                end_time = self.format_time(segment['end'])
                
                # Ensure minimum duration
                if segment['end'] - segment['start'] < 1.0:
                    end_time = self.format_time(segment['start'] + 1.0)
                
                # Smart line breaking for long text
                chinese_lines = self.smart_line_break(chinese_text)
                japanese_lines = self.smart_line_break(japanese_text)
                
                # Write SRT entry
                f.write(f"{i + 1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{chinese_lines}\n")
                f.write(f"{japanese_lines}\n\n")
                
                print(f"Processing progress: {i + 1}/{len(segments)}")
        
        print(f"Subtitle file generated: {output_path}")

    def smart_line_break(self, text, max_length=40):
        """
        Smart line breaking for subtitles
        """
        if len(text) <= max_length:
            return text
        
        # Try to break at punctuation first
        import re
        sentences = re.split(r'([。！？，、])', text)
        
        lines = []
        current_line = ""
        
        for part in sentences:
            if len(current_line + part) <= max_length:
                current_line += part
            else:
                if current_line:
                    lines.append(current_line.strip())
                current_line = part
        
        if current_line:
            lines.append(current_line.strip())
        
        return '\n'.join(lines[:2])  # Max 2 lines per subtitle
    
    def process_video_to_srt(self, video_path, output_dir=None):
        """
        Generate only Chinese subtitles SRT file, without burning.

        Return the path of the generated SRT file (success) or False (failure).
        """
        from pathlib import Path
        import uuid

        video_path = str(Path(video_path).resolve())
        if not video_path.lower().endswith((".mp4", ".mkv", ".avi", ".mov")):
            print(f"Error: {video_path} is not a supported video format")
            return False
        if not os.path.exists(video_path):
            print(f"Error: Video file does not exist: {video_path}")
            return False

        if output_dir is None:
            output_dir = os.path.dirname(video_path)
        output_dir = str(Path(output_dir).resolve())
        if not os.path.exists(output_dir):
            os.makedirs(output_dir)
        if not os.access(output_dir, os.W_OK):
            print(f"Error: No write permission: {output_dir}")
            return False

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(output_dir, f"{video_name}_{uuid.uuid4().hex}_temp.wav")
        subtitle_path = os.path.join(output_dir, f"{video_name}_chinese.srt")

        try:
            # Extract audio with progress
            if not self.extract_audio(video_path, audio_path):
                return False

            # Transcribe with progress callback
            print("Starting speech recognition...")
            transcription = self.transcribe_japanese(audio_path)

            if transcription is None or not transcription.get('segments'):
                print("Error: Speech recognition did not return a valid result")
                return False

            # Generate subtitles
            self.generate_srt(transcription, subtitle_path)

            # Clean up temp audio
            if os.path.exists(audio_path):
                os.remove(audio_path)

            print(f"SRT file generated: {subtitle_path}")
            return subtitle_path
        except Exception as e:
            import traceback
            print(f"Error occurred during processing: {e}\nStack trace: {traceback.format_exc()}")
            if os.path.exists(audio_path):
                print(f"Keep temporary audio file for debugging: {audio_path}")
            return False

    
