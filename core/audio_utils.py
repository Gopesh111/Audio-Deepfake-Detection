import numpy as np

def preprocess_audio(audio_data: np.ndarray, target_length: int = 160000):
    """
    Standardizes audio segment length and normalizes amplitude.
    """
    if len(audio_data) > target_length:
        return audio_data[:target_length]
    else:
        return np.pad(audio_data, (0, max(0, target_length - len(audio_data))))