import torch
import librosa
from transformers import Wav2Vec2Model, Wav2Vec2Processor

class Wav2VecFeatureExtractor:
    """
    Utilizes Huggingface's IndicWav2Vec to extract high-level audio features 
    for synthetic output detection.
    """
    def __init__(self, model_name: str = "ai4bharat/indicwav2vec-hindi"):
        self.processor = Wav2Vec2Processor.from_pretrained(model_name)
        self.model = Wav2Vec2Model.from_pretrained(model_name)
        self.model.eval()

    def extract_features(self, audio_path: str):
        # Load audio at 16kHz (Wav2Vec requirement)
        speech, _ = librosa.load(audio_path, sr=16000)
        
        # Preprocess and move to tensors
        inputs = self.processor(speech, sampling_rate=16000, return_tensors="pt", padding=True)
        
        with torch.no_grad():
            outputs = self.model(**inputs)
        
        # Use hidden states as feature embeddings
        return outputs.last_hidden_state.mean(dim=1)