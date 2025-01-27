import whisper
import ffmpeg
import os
import shutil  # Import for file operations
from docx import Document  # Import the library for Word document creation
import time  # Import time for measuring transcription duration
from tqdm import tqdm  # Import tqdm for progress display
from datetime import datetime  # Import datetime for folder naming
import logging
import json
import subprocess  # For system commands
from pyannote.audio import Pipeline # Import the pyannote library for speaker diarization

# Set up logging
log_file = "transcription_log.txt"
logging.basicConfig(
    filename=log_file,
    level=logging.INFO,
    format="%(asctime)s - %(levelname)s - %(message)s"
)

# Load dynamic configuration
CONFIG_FILE = "config.json"
def load_config():
    if not os.path.exists(CONFIG_FILE):
        # Default configuration if config file does not exist
        config = {
            "folder_path": "C:/path/to/default/folder",
            "model_size": "large", # Set to large for best results, but it will be slow without a CUDA-enabled GPU and CUDA setting below set to true
            "cuda_enabled": True, # Set to false if system does not have a CUDA-enabled GPU
            "transcription_params": {
                "language": "en",
                "temperature": 0.0,
                "compression_ratio_threshold": 2.4,
                "logprob_threshold": -0.5,
                "no_speech_threshold": 0.7,
                "condition_on_previous_text": False,
                "verbose": False
            },
            "disable_internet": True
        }
        with open(CONFIG_FILE, "w") as f:
            json.dump(config, f, indent=4)
        logging.info(f"Default configuration file created at {CONFIG_FILE}")
    else:
        with open(CONFIG_FILE, "r") as f:
            config = json.load(f)
    return config

# Function to disable internet
def disable_internet():
    try:
        logging.info("Disabling internet access.")
        subprocess.run(["ipconfig", "/release"], check=True, shell=True)
        logging.info("Internet successfully disabled.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to disable internet access: {e}")

# Function to enable internet
def enable_internet():
    try:
        logging.info("Enabling internet access.")
        subprocess.run(["ipconfig", "/renew"], check=True, shell=True)
        logging.info("Internet successfully enabled.")
    except subprocess.CalledProcessError as e:
        logging.error(f"Failed to enable internet access: {e}")

# Load the pretrained speaker diarization pipeline globally
diarization_pipeline = Pipeline.from_pretrained("pyannote/speaker-diarization")

def perform_diarization(audio_path):
    logging.info(f"Performing diarization for {audio_path}...")
    diarization_result = diarization_pipeline(audio_path)
    speaker_segments = []

    for turn, _, speaker in diarization_result.itertracks(yield_label=True):
        speaker_segments.append({
            "speaker": speaker,
            "start": turn.start,
            "end": turn.end,
        })
    return speaker_segments
    
# Load the Whisper model
def load_model(model_size, cuda_enabled):
    device = "cuda" if cuda_enabled else "cpu"
    return whisper.load_model(model_size, device=device)

def transcribe_file(file_path, model_size, output_folder, audio_output_folder, config):
    # Check for local ffmpeg and ffprobe binaries
    ffmpeg_executable = "./ffmpeg.exe" if os.path.exists("./ffmpeg.exe") else "ffmpeg"
    ffprobe_executable = "./ffprobe.exe" if os.path.exists("./ffprobe.exe") else "ffprobe"

    # Check if the file is a video or audio
    audio_path = f"{os.path.splitext(file_path)[0]}_processed.wav"

    # Process the file with FFmpeg for cleanup
    ffmpeg.input(file_path).output(audio_path, af="highpass=f=200, lowpass=f=3000", loglevel="error").run(overwrite_output=True)

    # Perform speaker diarization
    speaker_segments = perform_diarization(audio_path)

    # Load the Whisper model
    model = load_model(model_size, config["cuda_enabled"])

    # Get the duration of the audio file
    audio_info = ffmpeg.probe(audio_path)
    duration = float(audio_info['streams'][0]['duration'])

    # Transcribe audio
    logging.info(f"Transcribing audio for {file_path}...")
    start_time = time.time()  # Start the timer for measuring transcription duration

    progress_bar = tqdm(total=duration, desc="Transcribing", unit="s")

    result = model.transcribe(audio_path, **config["transcription_params"])

    progress_bar.close()
    end_time = time.time()  # End the timer
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    logging.info(f"Transcription completed for {file_path} in {hours} hours, {minutes} minutes.")
  
    # Add speaker labels to transcription segments
    labeled_segments = []
    for segment in result["segments"]:
        for speaker_segment in speaker_segments:
            if speaker_segment["start"] <= segment["start"] < speaker_segment["end"]:
                labeled_segments.append({
                    "speaker": speaker_segment["speaker"],
                    "start": segment["start"],
                    "end": segment["end"],
                    "text": segment["text"],
                })
                break
        else:
            labeled_segments.append(segment)  # Keep the original segment if no match

    result["segments"] = labeled_segments
    result["speaker_segments"] = speaker_segments

    # Add metadata to the transcription result
    result['metadata'] = {
        "file_path": file_path,
        "transcription_duration": f"{hours}h {minutes}m {seconds}s",
        "model": model_size,
        "device": "cuda" if config["cuda_enabled"] else "cpu",
        **config["transcription_params"],
        "warning": "Notice: This transcription may contain inaccuracies. Variations in word recognition, repetition of text during periods of silence, and occasional omissions or additions are possible, as the system prioritizes capturing faint speech over excluding it. If you have any concerns, questions, or feedback, please contact Travis Martin at travis.martin@mspd.mo.gov."
    }

    # Insert warning at the top of the transcription
    result["segments"].insert(0, {
    "start": 0.0,
    "end": 0.0,
    "text": f"{result['metadata']['warning']}"
    })

    # Clean up temporary audio file
    os.remove(audio_path)

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

    # Add metadata to the bottom of the document
    document.add_heading("Metadata", level=2)
    for key, value in transcription_result.get("metadata", {}).items():
        document.add_paragraph(f"{key}: {value}")

    # Save the document in the transcripts folder
    output_word_file = os.path.join(output_folder, f"{os.path.basename(os.path.splitext(file_path)[0])}_transcription.docx")
    document.save(output_word_file)
    logging.info(f"Transcription saved to {output_word_file}")

def format_time(seconds):
    minutes, seconds = divmod(int(seconds), 60)
    return f"{int(minutes):02}:{int(seconds):02}"

def batch_transcribe():
    # Load configuration
    config = load_config()
    folder_path = "C:/transcription-files"

    model_size = config["model_size"]

    logging.info(f"Using model: {model_size}")

    # Disable internet if configured
    if config.get("disable_internet", False):
        disable_internet()

    # Create a new folder for transcripts
    timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
    output_folder = os.path.join(folder_path, f"transcripts-{timestamp}")
    os.makedirs(output_folder, exist_ok=True)

    files = []
    for root, _, file_list in os.walk(folder_path):
        for file in file_list:
            if file.endswith((".mp4", ".avi", ".mkv", ".mov", ".wav", ".mp3", ".aac", ".flac")):
                files.append(os.path.join(root, file))

    for file_path in tqdm(files, desc="Processing Files"):
        # Transcribe the file
        transcription_result = transcribe_file(file_path, model_size, output_folder, None, config)

        # Save to Word document in the transcripts folder
        save_to_word(transcription_result, file_path, output_folder)

    # Re-enable internet if it was disabled
    if config.get("disable_internet", False):
        enable_internet()

if __name__ == "__main__":
    batch_transcribe()
    logging.info("Batch transcription completed.")
