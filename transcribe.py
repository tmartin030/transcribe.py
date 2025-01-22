import whisper
import ffmpeg
import os
import json  # Import the JSON library
from docx import Document  # Import the library for Word document creation

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

def save_to_word_with_hyperlinks(transcription_result, video_file):
    document = Document()
    document.add_heading("Transcription", level=1)

    for segment in transcription_result.get("segments", []):
        start_time = segment["start"]
        end_time = segment["end"]
        text = segment["text"]

        # Format timestamps and create a hyperlink-like format
        timestamp = f"[{format_time(start_time)} - {format_time(end_time)}]"
        paragraph = document.add_paragraph()
        paragraph.add_run(timestamp).bold = True
        paragraph.add_run(f" {text}")

    # Save the document
    output_word_file = os.path.splitext(video_file)[0] + "_transcription.docx"
    document.save(output_word_file)
    print(f"Transcription with hyperlinks saved to {output_word_file}")

def format_time(seconds):
    hours, remainder = divmod(seconds, 3600)
    minutes, seconds = divmod(remainder, 60)
    return f"{int(hours):02}:{int(minutes):02}:{int(seconds):02}"

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

        # Save to Word document with hyperlinks
        save_to_word_with_hyperlinks(transcription_result, video_file)
