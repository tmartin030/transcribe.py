import whisper
import ffmpeg
import os
import json  # Import the JSON library

# Load the Whisper model
model = whisper.load_model("base")  # Choose from: tiny, base, small, medium, large

def transcribe_video(video_path):
    # Extract audio from video
    audio_path = "audio.wav"
    ffmpeg.input(video_path).output(audio_path, format="wav").run(overwrite_output=True)
    
    # Transcribe audio
    print("Transcribing audio...")
    result = model.transcribe(audio_path)
    
    # Clean up temporary audio file
    os.remove(audio_path)

    # Return the full result for JSON output
    return result

if __name__ == "__main__":
    video_file = input("Enter the path to the video file: ")
    if not os.path.exists(video_file):
        print("File not found!")
    else:
        # Transcribe the video
        transcription_result = transcribe_video(video_file)

        # Save the transcription result as a JSON file
        output_file = os.path.splitext(video_file)[0] + "_transcription.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcription_result, f, indent=4, ensure_ascii=False)

        print(f"Transcription saved to {output_file}")
