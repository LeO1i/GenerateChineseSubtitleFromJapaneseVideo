from speech_extract import JapaneseVideoSubtitleGenerator
from write_sutitle import WriteSubtitle
import os
import os.path


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
        output_dir = os.path.dirname(srt_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, f"{video_name}_cn_hardsub.mp4")
        if writer.burn_subtitles(video_path, srt_path, output_video_path):
            print("Burned successfully.")
            return -1
        else:
            print("Burn failed. Please check errors above.")
            return -1
    else:
        pass

    # Use Instance
    generator = JapaneseVideoSubtitleGenerator()
    # Specify the path of the Japanese video file
    video_path = input("Please input the path of the Japanese video file: ").strip()

    # Generate SRT
    srt_path = generator.process_video(video_path)

    if not srt_path:
        print("Subtitle generation failed, please check the error information.")
        return

    print("Open the SRT file, review or edit it if needed.")
    choice = input("Do you want to burn this subtitle into the video now? (y/N): ").strip().lower()

    if choice == 'y':
        output_dir = os.path.dirname(srt_path)
        video_name = os.path.splitext(os.path.basename(video_path))[0]
        output_video_path = os.path.join(output_dir, f"{video_name}_cn_hardsub.mp4")
        if writer.burn_subtitles(video_path, srt_path, output_video_path):
            print("Burned successfully.")
        else:
            print("Burn failed. Please check errors above.")
    else:
        print("Skipped burning. You can run again to burn after editing the SRT.")

if __name__ == "__main__":
    main()