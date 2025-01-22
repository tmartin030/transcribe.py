import whisper
import ffmpeg
import os
import json  # Import the JSON library
from docx import Document  # Import the library for Word document creation
import time  # Import time for measuring transcription duration

# Load the Whisper model
model = whisper.load_model("large", device="cuda")  # Choose from: tiny, base, small, medium, large

def transcribe_video(video_path):
    # Extract audio from video
    audio_path = "audio.wav"
    # Perform audio cleanup (optional, but recommended) for noisy audio, narrow the range (e.g., highpass=f=250, lowpass=f=2500).
    ffmpeg.input(video_path).output(audio_path, af="highpass=f=200, lowpass=f=3000").run(overwrite_output=True)
    # Extract audio without cleanup (alternative if no audio cleanup is needed)
    # ffmpeg.input(video_path).output(audio_path, format="wav").run(overwrite_output=True)
    
    # Transcribe audio
    print("Transcribing audio...")
    start_time = time.time()  # Start the timer for measuring transcription duration
    result = model.transcribe(
        audio_path,
        language="en",  # Specify language explicitly, helps to improve transcription accuracy
        temperature=0.0, # Set temperature to 0.0 for best results. A value of 0.0 means the model will take the most likely prediction at each step, minimizing variability. Higher values introduce more creative or diverse results but may reduce accuracy.
        compression_ratio_threshold=2.4, # This helps handle text with high compression ratios (e.g., gibberish or highly repetitive text). If the generated text exceeds this ratio, it may be discarded to ensure quality. Lower this value if you're getting overly compressed outputs.
        logprob_threshold=-1.0, # Set the log probability threshold. A lower value will increase the number of words transcribed but may also increase the number of errors. A higher value will reduce the number of words transcribed but may also reduce the number of errors.
        no_speech_threshold=0.3 # Set the threshold for no speech detection. A higher value will reduce the number of false positives but may also reduce the
    )
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Transcription completed in {hours} hours, {minutes} minutes.")

    # Clean up temporary audio file
    os.remove(audio_path)

    # Return the full result for JSON output
    return result

def save_to_word(transcription_result, video_file):
    document = Document()
    document.add_heading("Transcription", level=1)

    for segment in transcription_result.get("segments", []):
        start_time = segment["start"]
        text = segment["text"]

        # Format timestamps and create a hyperlink-like format
        timestamp = f"[{format_time(start_time)}]"
        paragraph = document.add_paragraph()
        paragraph.add_run(timestamp).bold = True
        paragraph.add_run(f" {text}")

    # Save the document
    output_word_file = os.path.splitext(video_file)[0] + "_transcription.docx"
    document.save(output_word_file)
    print(f"Transcription saved to {output_word_file}")

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{int(minutes):02}:{int(seconds):02}"

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
        save_to_word(transcription_result, video_file)