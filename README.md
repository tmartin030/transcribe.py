# transcribe.py
Users may change the program's behavior by editing the JSON file. The available parameters are described below:

folder_path: Path to the folder containing audio or video files to process. If this is not changed, default path is C:/transcription-files

model_size: Specifies the Whisper model to use. Options are "tiny", "base", "small", "medium", or "large". Larger models are more accurate but require more resources. Use large for best results, but beware it will take longer if cuda_enabled (below) is set to false due to the fact the system does not have a CUDA-enabled GPU.

cuda_enabled: Set to true if the system has a CUDA-enabled GPU for faster processing. Otherwise, set to false for CPU processing.

transcription_params:
  - language: 
        Specifies the language of the audio (e.g., "en" for English).
  - temperature: 
        Range: 0.0 to 1.0. 
        
        Description: Controls the randomness of predictions. Lower values (like the default 0.0) prioritize accuracy, while higher values introduce variability and creativity. 
        
        Recommendation: Default is 0.0; however, we should experiment with this since the program is essentially returning high-probability words when we're dealing with low-probability events. Not sure if that logic applies here, but try it and report back.

  - compression_ratio_threshold: 
        Range: Positive values, typically between 1.0 and 3.0. 
        
        Description: Helps filter overly compressed outputs (e.g., gibberish). Lower values are stricter, discarding more low-quality text. A value of 2.4 is generally a good balance.

        Recommendation: Default is 2.4, which works well for most cases.
  - logprob_threshold: 
        Range: Negative values (e.g., -2.0 to 0.0).
        
        Description: Sets a confidence threshold for word predictions. A lower (more negative) value allows more words but may increase errors. Higher values reduce errors but may exclude valid words.
        
        Recommendation: Default to -1.0 to -0.5 for balancing inclusion and accuracy.
  
  - no_speech_threshold: 
        Range: 0.0 to 1.0
        
        Description: Determines sensitivity to silence. Higher values (closer to 1.0) reduce false positives (erroneous speech detection) but may miss faint speech. Lower values increase sensitivity but risk more false positives.
        
        Recommendation: Default to 0.7 for a good trade-off between sensitivity and accuracy.
  - condition_on_previous_text: 
        Options: True or False
        
        Description: If True, predictions are conditioned on previously transcribed text for context. This can improve coherence but may introduce errors in segments with poor initial transcription.
        
        Recommendation: Default to False to avoid compounding errors.

  - verbose: 
    Options: True or False
    
    Description: If True, provides detailed logs during transcription. Useful for debugging or understanding model behavior.
    
    Recommendation: Default to False. Note: Don't change this one. It won't hurt anything, I guess. But who knows.

disable_internet: 
    If true, disables internet access during transcription for privacy and security. Keep this true for now.


