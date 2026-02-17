from speech_extract import JapaneseVideoSubtitleGenerator
from write_sutitle import WriteSubtitle
import os


def _build_hardsub_path(video_path, output_dir):
    """Build the output path for a hardsubbed video."""
    video_name = os.path.splitext(os.path.basename(video_path))[0]
    return os.path.join(output_dir, f"{video_name}_cn_hardsub.mp4")


def main():

    print("Welcome to the Japanese video subtitle generator!")
    print("Do you already have a Japanese subtitle file generate by this program?")
    choice = input("Please input y or n: ").strip().lower()
    writer = WriteSubtitle()
    if choice == 'y':
        print("Please input the path of the Japanese subtitle file: ")
        srt_path = input().strip()
        print("Please input the path of the Japanese video file: ")
        video_path = input().strip()
        output_video_path = _build_hardsub_path(video_path, os.path.dirname(srt_path))
        if writer.burn_subtitles(video_path, srt_path, output_video_path):
            print("Burned successfully.")
            return 0
        else:
            print("Burn failed. Please check errors above.")
            return 1
    else:
        pass

    asr_model = input("ASR model (default Qwen/Qwen3-ASR-1.7B): ").strip() or "Qwen/Qwen3-ASR-1.7B"
    mt_model = input("MT model (default tencent/HY-MT1.5-1.8B): ").strip() or "tencent/HY-MT1.5-1.8B"
    use_advanced_mt = input("Enable advanced MT first (HY-MT 7B style fallback)? (y/N): ").strip().lower() == "y"
    quality_mode = input("Quality mode fast/accurate (default fast): ").strip().lower() or "fast"
    if quality_mode not in {"fast", "accurate"}:
        quality_mode = "fast"
    chunk_size_text = input("Chunk size in seconds (30-600, default 120): ").strip() or "120"
    overlap_text = input("Chunk overlap seconds (0-10, default 1.5): ").strip() or "1.5"
    glossary_path = input("Glossary file path (optional, press Enter to skip): ").strip() or None

    try:
        chunk_size = int(chunk_size_text)
        if chunk_size < 30 or chunk_size > 600:
            raise ValueError
    except Exception:
        print("Invalid chunk size. Using default 120.")
        chunk_size = 120

    try:
        overlap_seconds = float(overlap_text)
        if overlap_seconds < 0 or overlap_seconds > 10:
            raise ValueError
    except Exception:
        print("Invalid overlap. Using default 1.5.")
        overlap_seconds = 1.5

    if glossary_path and not os.path.exists(glossary_path):
        print("Glossary file not found. Ignoring glossary setting.")
        glossary_path = None

    # Use Instance
    generator = JapaneseVideoSubtitleGenerator(
        asr_model_id=asr_model,
        mt_model_id=mt_model,
        use_advanced_mt=use_advanced_mt,
        quality_mode=quality_mode,
        glossary_path=glossary_path,
    )
    # Specify the path of the Japanese video file
    video_path = input("Please input the path of the Japanese video file: ").strip()

    # Generate SRT
    srt_path = generator.process_video(
        video_path,
        chunk_size_seconds=chunk_size,
        overlap_seconds=overlap_seconds,
        quality_mode=quality_mode,
        glossary_path=glossary_path,
    )

    if not srt_path:
        print("Subtitle generation failed, please check the error information.")
        return

    print("Open the SRT file, review or edit it if needed.")
    choice = input("Do you want to burn this subtitle into the video now? (y/N): ").strip().lower()

    if choice == 'y':
        output_video_path = _build_hardsub_path(video_path, os.path.dirname(srt_path))
        if writer.burn_subtitles(video_path, srt_path, output_video_path):
            print("Burned successfully.")
        else:
            print("Burn failed. Please check errors above.")
    else:
        print("Skipped burning. You can run again to burn after editing the SRT.")

if __name__ == "__main__":
    main()