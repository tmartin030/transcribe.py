import whisper
import ffmpeg
import os
import shutil  # Import for file operations
from docx import Document  # Import the library for Word document creation
import time  # Import time for measuring transcription duration
from tqdm import tqdm  # Import tqdm for progress display
from datetime import datetime  # Import datetime for folder naming

# Centralized transcription parameters
TRANSCRIPTION_PARAMS = {
    "language": "en",
    "temperature": 0.0,
    "compression_ratio_threshold": 2.4,
    "logprob_threshold": -0.5,
    "no_speech_threshold": 0.7,
    "condition_on_previous_text": False,
    "verbose": False
}

# Load the Whisper model
def load_model(model_size):
    return whisper.load_model(model_size, device="cuda")  # Choose from: tiny, base, small, medium, large

def transcribe_file(file_path, model_size, output_folder, audio_output_folder):
    # Check if the file is a video or audio
    audio_path = f"{os.path.splitext(file_path)[0]}_processed.wav"

    # Process the file with FFmpeg for cleanup
    ffmpeg.input(file_path).output(audio_path, af="highpass=f=200, lowpass=f=3000", loglevel="error").run(overwrite_output=True)

    # Copy the processed audio file to the processed audio folder
    processed_audio_copy = os.path.join(audio_output_folder, f"{os.path.basename(os.path.splitext(file_path)[0])}_processed_copy.wav")
    shutil.move(audio_path, processed_audio_copy)
    print(f"Processed audio file saved as a copy to {processed_audio_copy}")

    # Load the Whisper model
    model = load_model(model_size)

    # Get the duration of the audio file
    audio_info = ffmpeg.probe(processed_audio_copy)
    duration = float(audio_info['streams'][0]['duration'])

    # Transcribe audio
    print(f"Transcribing audio for {file_path}...")
    start_time = time.time()  # Start the timer for measuring transcription duration

    progress_bar = tqdm(total=duration, desc="Transcribing", unit="s")

    result = model.transcribe(processed_audio_copy, **TRANSCRIPTION_PARAMS)

    progress_bar.close()
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Transcription completed for {file_path} in {hours} hours, {minutes} minutes.")

    # Add metadata to the transcription result
    result['metadata'] = {
        "file_path": file_path,
        "transcription_duration": f"{hours}h {minutes}m {seconds}s",
        "model": model_size,
        "device": "cuda",
        **TRANSCRIPTION_PARAMS,
        "warning": "This transcription may contain inaccuracies, including but not limited to: repetition where none exists, omission of text when speaking occurred, inclusion of text where speaking did not occur, and other transcription problems."
    }

    # Insert metadata and warning at the top of the transcription
    result["segments"].insert(0, {
    "start": 0.0,
    "end": 0.0,
    "text": (
        f"Warning: {result['metadata']['warning']}\n\n"
        f"Metadata:\n"
        f"File Path: {result['metadata']['file_path']}\n"
        f"Transcription Duration: {result['metadata']['transcription_duration']}\n"
        f"Model: {result['metadata']['model']}\n"
        f"Device: {result['metadata']['device']}\n"
        f"Language: {result['metadata']['language']}\n"
        f"Temperature: {result['metadata']['temperature']}\n"
        f"Compression Ratio Threshold: {result['metadata']['compression_ratio_threshold']}\n"
        f"Logprob Threshold: {result['metadata']['logprob_threshold']}\n"
        f"No Speech Threshold: {result['metadata']['no_speech_threshold']}\n"
        f"Condition on Previous Text: {result['metadata']['condition_on_previous_text']}"
    )
})

    return result

def save_to_word(transcription_result, file_path, output_folder):
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

    # Save the document in the transcripts folder
    output_word_file = os.path.join(output_folder, f"{os.path.basename(os.path.splitext(file_path)[0])}_transcription.docx")
    document.save(output_word_file)
    print(f"Transcription saved to {output_word_file}")

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{int(minutes):02}:{int(seconds):02}"

def batch_transcribe(folder_path):
    print("Select the Whisper model size for this batch (enter the number):")
    print("1. tiny\n2. base\n3. small (default)\n4. medium\n5. large")
    model_size_map = {"1": "tiny", "2": "base", "3": "small", "4": "medium", "5": "large"}

    while True:
        choice = input("Enter the number corresponding to your choice: ")
        if choice in model_size_map:
            model_size = model_size_map[choice]
            break
        print("Invalid choice. Please enter a number between 1 and 5.")

    print(f"Using model: {model_size}")

    # Create a new folder for transcripts
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(folder_path, f"transcripts-{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    # Create a new folder for processed audio files
    audio_output_folder = os.path.join(folder_path, f"transcript-audio-{timestamp}")
    os.makedirs(audio_output_folder, exist_ok=True)

    files = []
    for root, _, file_list in os.walk(folder_path):
        for file in file_list:
            if file.endswith(('.mp4', '.avi', '.mkv', '.mov', '.wav', '.mp3', '.aac', '.flac', '.ps', '.psx')):
                files.append(os.path.join(root, file))

    for file_path in tqdm(files, desc="Processing Files"):
        # Convert proprietary formats to .wav
        if file_path.endswith(('.ps', '.psx')):
            converted_file_path = os.path.splitext(file_path)[0] + "_converted.wav"
            print(f"Converting {file_path} to {converted_file_path}...")
            ffmpeg.input(file_path).output(converted_file_path).run(overwrite_output=True)
            file_path = converted_file_path  # Update file_path to point to the converted file

        # Transcribe the file
        transcription_result = transcribe_file(file_path, model_size, output_folder, audio_output_folder)

        # Save to Word document in the transcripts folder
        save_to_word(transcription_result, file_path, output_folder)

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing audio or video files: ")
    if not os.path.exists(folder_path):
        print("Folder not found!")
    else:
        batch_transcribe(folder_path)
