import whisper
import ffmpeg
import os
import json  # Import the JSON library
from docx import Document  # Import the library for Word document creation
import time  # Import time for measuring transcription duration
from tqdm import tqdm  # Import tqdm for progress display

# Load the Whisper model
def load_model():
    return whisper.load_model("medium", device="cuda")  # Choose from: tiny, base, small, medium, large

def transcribe_file(file_path):
    # Check if the file is a video or audio
    audio_path = f"{os.path.splitext(file_path)[0]}_processed.wav"

    # Process the file with FFmpeg for cleanup
    ffmpeg.input(file_path).output(audio_path, af="highpass=f=200, lowpass=f=3000").run(overwrite_output=True)

    # Load the Whisper model
    model = load_model()

    # Get the duration of the audio file
    audio_info = ffmpeg.probe(audio_path)
    duration = float(audio_info['streams'][0]['duration'])

    # Transcribe audio
    print(f"Transcribing audio for {file_path}...")
    start_time = time.time()  # Start the timer for measuring transcription duration

    progress_bar = tqdm(total=duration, desc="Transcribing", unit="s")

    def progress_callback(segment):
        progress_bar.update(segment["end"] - segment["start"])

    result = model.transcribe(
        audio_path,
        language="en",  # Specify language explicitly, helps to improve transcription accuracy
        temperature=0.0, # Set temperature to 0.0 for best results. A value of 0.0 means the model will take the most likely prediction at each step, minimizing variability. Higher values introduce more creative or diverse results but may reduce accuracy.
        compression_ratio_threshold=2.4, # Handle text with high compression ratios (e.g., gibberish or highly repetitive text).
        logprob_threshold=-1.0, # Set the log probability threshold to balance transcription quality and errors.
        no_speech_threshold=0.4, # Set the threshold for no speech detection.
        condition_on_previous_text=False, # Disable conditioning on previous text to prevent repetitive outputs.
        verbose=False
    )

    progress_bar.close()
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Transcription completed for {file_path} in {hours} hours, {minutes} minutes.")

    # Clean up temporary audio file
    os.remove(audio_path)

    # Add metadata to the transcription result
    result['metadata'] = {
        "file_path": file_path,
        "transcription_duration": f"{hours}h {minutes}m {seconds}s",
        "model": "medium",
        "device": "cuda",
        "language": "en",
        "temperature": 0.0,
        "compression_ratio_threshold": 2.4,
        "logprob_threshold": -1.0,
        "no_speech_threshold": 0.4,
        "condition_on_previous_text": False
    }

    # Adjust speaker breaks for longer delays
    merged_segments = []
    current_segment = None

    for segment in result["segments"]:
        if current_segment is None:
            current_segment = segment
        else:
            # Check if segments are close enough to merge
            if segment["start"] - current_segment["end"] < .15:  # decreased gap to .5 seconds before splitting
                current_segment["text"] += " " + segment["text"]
                current_segment["end"] = segment["end"]
            else:
                merged_segments.append(current_segment)
                current_segment = segment

    if current_segment:
        merged_segments.append(current_segment)

    result["segments"] = merged_segments

    return result

def save_to_word(transcription_result, file_path):
    document = Document()
    document.add_heading("Transcription", level=1)

    for segment in transcription_result.get("segments", []):
        start_time = segment["start"]
        text = segment["text"]

        # Format timestamps and create a hyperlink-like format
        timestamp = f"[{format_time(start_time)}]"
        paragraph = document.add_paragraph()
        paragraph.add_run(timestamp).italic = True
        paragraph.add_run(f" {text}")

    # Add metadata to the document
    document.add_heading("Metadata", level=2)
    for key, value in transcription_result.get("metadata", {}).items():
        document.add_paragraph(f"{key}: {value}")

    # Save the document
    output_word_file = os.path.splitext(file_path)[0] + "_transcription.docx"
    document.save(output_word_file)
    print(f"Transcription saved to {output_word_file}")

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{int(minutes):02}:{int(seconds):02}"

def batch_transcribe(folder_path):
    files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.endswith(('.mp4', '.avi', '.mkv', '.mov', '.wav', '.mp3', '.aac', '.flac'))
    ]

    for file_path in tqdm(files, desc="Processing Files"):
        # Transcribe the file
        transcription_result = transcribe_file(file_path)

        # Save the transcription result as a JSON file
        output_file = os.path.splitext(file_path)[0] + "_transcription.json"
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(transcription_result, f, indent=4, ensure_ascii=False)

        print(f"Transcription saved to {output_file}")

        # Save to Word document with hyperlinks
        save_to_word(transcription_result, file_path)

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing audio or video files: ")
    if not os.path.exists(folder_path):
        print("Folder not found!")
    else:
        batch_transcribe(folder_path)