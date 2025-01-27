# transcribe.py
whisperai video transcription with json output
This is a first attempt at video transcription using whisperai. 
Must use python 3.11 (not 3.12+) as of 1/22/25

** this may be redundant if pip install ffmpeg-python works ** Install FFMPEG by downloading the 7z file and extracting c:\ffmpeg

Install the following using terminal before running:
pip install git+https://github.com/openai/whisper.git
pip install python-docx
pip install ffmpeg-python

Ensure python and ffmpeg are added to PATH

if using GPU with CUDA (assuming CUDA v12.1+): 
    pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu121

    modify code 


