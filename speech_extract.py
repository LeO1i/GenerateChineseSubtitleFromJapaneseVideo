import os
import subprocess
import whisper
import datetime
import torch
from pathlib import Path
import uuid
import time
from googletrans import Translator

class JapaneseVideoSubtitleGenerator:
    def __init__(self, model_name="small", device=None):
        self.device = device or ("cuda:0" if torch.cuda.is_available() else "cpu")
        print(f"Loading Whisper model: {model_name} on {self.device}...")
        self.whisper_model = whisper.load_model(model_name, device=self.device)
        self.translator = Translator()

    def extract_audio(self, video_path, audio_path):
        """Extract audio from video with noise reduction"""
        print("Extracting audio from video...")
        try:
            cmd = [
                'ffmpeg',
                '-i', video_path,
                '-acodec', 'pcm_s16le',
                '-ar', '16000',
                '-ac', '1',
                '-af', 'highpass=f=80,lowpass=f=8000,volume=1.5,dynaudnorm',
                audio_path, '-y'
            ]
            subprocess.run(cmd, check=True, capture_output=True)
            print(f"Audio extraction completed: {audio_path}")
            return True
        except subprocess.CalledProcessError as e:
            print(f"Audio extraction failed: {e}")
            return False
        except FileNotFoundError:
            # Fallback to a common Windows ffmpeg path if not in PATH
            try:
                cmd = [
                    'C:/ffmpeg/ffmpeg-master-latest-win64-gpl/bin/ffmpeg.exe',
                    '-i', video_path,
                    '-acodec', 'pcm_s16le',
                    '-ar', '16000',
                    '-ac', '1',
                    '-af', 'highpass=f=80,lowpass=f=8000,volume=1.5,dynaudnorm',
                    audio_path, '-y'
                ]
                subprocess.run(cmd, check=True, capture_output=True)
                print(f"Audio extraction completed: {audio_path}")
                return True
            except Exception as e:
                print(f"Audio extraction failed (ffmpeg not found): {e}")
                return False

    def transcribe_japanese(self, audio_path):
        """Perform Japanese speech recognition"""
        print("Starting Japanese speech recognition...")
        try:
            result = self.whisper_model.transcribe(
                audio_path,
                language='ja',
                task='transcribe',
                word_timestamps=True,
                temperature=0.0,
                beam_size=5,
                best_of=5,
                patience=1.0,
                condition_on_previous_text=True,
                compression_ratio_threshold=2.4,
                logprob_threshold=-1.0,
                no_speech_threshold=0.6
            )
            print("Japanese speech recognition completed")
            return result
        except Exception as e:
            print(f"Speech recognition failed: {e}")
            return None

    def format_time(self, seconds):
        """Convert seconds to SRT format time"""
        td = datetime.timedelta(seconds=seconds)
        hours = int(td.total_seconds() // 3600)
        minutes = int((td.total_seconds() % 3600) // 60)
        seconds = td.total_seconds() % 60
        milliseconds = int((seconds % 1) * 1000)
        seconds = int(seconds)
        return f"{hours:02d}:{minutes:02d}:{seconds:02d},{milliseconds:03d}"

    def process_segments(self, transcription_result):
        """Process and validate transcription segments"""
        processed_segments = []
        for segment in transcription_result['segments']:
            if segment['end'] - segment['start'] < 0.5:
                continue
            if segment.get('confidence', 1.0) < 0.3:
                continue
            
            if (processed_segments and 
                segment['start'] - processed_segments[-1]['end'] < 1.0 and
                len(segment['text'].strip()) < 10):
                processed_segments[-1]['text'] += ' ' + segment['text']
                processed_segments[-1]['end'] = segment['end']
            else:
                processed_segments.append(segment)
        return processed_segments

    def translate_text(self, text):
        """Translate Japanese text to Chinese using Google Translate"""
        try:
            result = self.translator.translate(text, src='ja', dest='zh-cn')
            return result.text.strip()
        except Exception as e:
            print(f"Translation error: {e}")
            return text

    def generate_bilingual_srt(self, transcription_result, output_path):
        """Generate SRT file with both Japanese and Chinese subtitles"""
        print("Generating bilingual subtitles...")
        segments = self.process_segments(transcription_result)
        
        with open(output_path, 'w', encoding='utf-8') as f:
            for i, segment in enumerate(segments):
                japanese_text = segment['text'].strip()
                if not japanese_text:
                    continue

                chinese_text = self.translate_text(japanese_text)
                start_time = self.format_time(max(0, segment['start']))
                end_time = self.format_time(segment['end'])

                f.write(f"{i + 1}\n")
                f.write(f"{start_time} --> {end_time}\n")
                f.write(f"{japanese_text}\n{chinese_text}\n\n")
                
                print(f"Processing progress: {i + 1}/{len(segments)}")
        

    def process_video(self, video_path, output_dir=None):
        """Main function to process video and generate bilingual subtitles"""
        video_path = str(Path(video_path).resolve())
        if not video_path.lower().endswith((".mp4", ".mkv", ".avi", ".mov", ".wmv", ".flv")):
            print(f"Error: {video_path} is not a supported video format")
            return False

        if output_dir is None:
            output_dir = os.path.dirname(video_path)
        output_dir = str(Path(output_dir).resolve())
        os.makedirs(output_dir, exist_ok=True)

        video_name = os.path.splitext(os.path.basename(video_path))[0]
        audio_path = os.path.join(output_dir, f"{video_name}_{uuid.uuid4().hex}_temp.wav")
        subtitle_path = os.path.join(output_dir, f"{video_name}_bilingual.srt")

        try:
            if not self.extract_audio(video_path, audio_path):
                return False

            transcription = self.transcribe_japanese(audio_path)
            if transcription is None or not transcription.get('segments'):
                print("Error: Speech recognition failed")
                return False

            self.generate_bilingual_srt(transcription, subtitle_path)

            if os.path.exists(audio_path):
                os.remove(audio_path)

            print(f"Processing completed. Subtitle file: {subtitle_path}")
            return subtitle_path

        except Exception as e:
            print(f"Error during processing: {e}")
            if os.path.exists(audio_path):
                os.remove(audio_path)
            return False

    def process_video_to_srt(self, video_path, output_dir=None):
        """Compatibility wrapper used by GUI to generate SRT and return its path"""
        return self.process_video(video_path, output_dir)