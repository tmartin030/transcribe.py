import whisper
import ffmpeg
import os
import json  # Import the JSON library
from docx import Document  # Import the library for Word document creation
import time  # Import time for measuring transcription duration
from tqdm import tqdm  # Import tqdm for progress display
from multiprocessing import Pool, cpu_count, Manager  # Import for parallel processing

# Function to load the Whisper model per process
def get_model():
    return whisper.load_model("medium", device="cuda")

def transcribe_audio(audio_path, progress_dict, index):
    # Load the model per process
    model = get_model()

    # Transcribe audio
    print(f"Starting transcription for {audio_path}...")  # Diagnostic log
    start_time = time.time()

    result = model.transcribe(
        audio_path,
        language="en",  # Specify language explicitly
        temperature=0.0,
        compression_ratio_threshold=2.4,  # Tolerate slightly higher compression
        logprob_threshold=-1.0,
        no_speech_threshold=0.4,  # Reduce sensitivity to short silences
        condition_on_previous_text=True  # Enable context-based transcription
    )

    # Merge small segments manually (post-processing)
    merged_segments = []
    current_segment = None

    for segment in result["segments"]:
        if current_segment is None:
            current_segment = segment
        else:
            # Check if segments are close enough to merge
            if segment["start"] - current_segment["end"] < 1.0:  # Merge if less than 1 second gap
                current_segment["text"] += " " + segment["text"]
                current_segment["end"] = segment["end"]
            else:
                merged_segments.append(current_segment)
                current_segment = segment

    if current_segment:
        merged_segments.append(current_segment)

    result["segments"] = merged_segments

    end_time = time.time()
    elapsed_time = end_time - start_time
    hours, remainder = divmod(int(elapsed_time), 3600)
    minutes, seconds = divmod(remainder, 60)
    print(f"Transcription completed for {audio_path} in {hours} hours, {minutes} minutes.")

    # Add metadata to the transcription result
    if 'metadata' not in result:
        result['metadata'] = {}
    result['metadata'].update({
        "audio_path": audio_path,
        "transcription_duration": f"{hours}h {minutes}m {seconds}s",
        "model": "medium",
        "device": "cuda",
        "language": "en"
    })

    # Save transcription result as a JSON file
    output_file = os.path.splitext(audio_path)[0] + "_transcription.json"
    with open(output_file, "w", encoding="utf-8") as f:
        json.dump(result, f, indent=4, ensure_ascii=False)
    print(f"Transcription saved to {output_file}")

    # Update progress dictionary
    progress_dict[index] = True
    return result

def process_file(args):
    file_path, progress_dict, index = args
    try:
        print(f"Processing file: {file_path}")  # Diagnostic log
        # Check if it's an audio or video file and clean it up with ffmpeg
        audio_path = f"{os.path.splitext(file_path)[0]}_processed_audio.wav"
        ffmpeg.input(file_path).output(audio_path, af="highpass=f=200, lowpass=f=3000").run(overwrite_output=True)

        result = transcribe_audio(audio_path, progress_dict, index)

        # Clean up temporary audio file
        os.remove(audio_path)

        save_to_word(result, file_path)
        print(f"Finished processing: {file_path}")  # Diagnostic log
    except Exception as e:
        print(f"Error processing {file_path}: {e}")

def save_to_word(transcription_result, file_path):
    document = Document()
    document.add_heading("Transcription", level=1)

    for segment in transcription_result.get("segments", []):
        start_time = segment["start"]
        text = segment["text"]

        # Format timestamps and text
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

    with Manager() as manager:
        progress_dict = manager.dict()  # Use a dictionary to track progress
        total_files = len(files)

        # Initialize progress dictionary
        for index in range(total_files):
            progress_dict[index] = False

        # Use multiprocessing for parallel transcription
        max_processes = 2
        with Pool(processes=max_processes) as pool:
            for _ in tqdm(pool.imap(process_file, [(file, progress_dict, idx) for idx, file in enumerate(files)]), total=total_files, desc="Processing Files"):
                pass

        completed_files = sum(progress_dict.values())
        print(f"All files processed. Total: {completed_files}/{total_files}.")  # Final progress log

if __name__ == "__main__":
    folder_path = input("Enter the path to the folder containing audio or video files: ")
    if not os.path.exists(folder_path):
        print("Folder not found!")
    else:
        batch_transcribe(folder_path)
