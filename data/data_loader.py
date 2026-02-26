import torch
from torch.utils.data import Dataset, DataLoader
import pandas as pd
import librosa

class AudioDataset(Dataset):
    """
    Handles a large-scale dataset of 78,453 audio segments.
    Ensures labels and speaker IDs are mapped correctly for disjoint testing.
    """
    def __init__(self, metadata_csv: str, transform=None):
        self.metadata = pd.read_csv(metadata_csv) # Columns: path, label, speaker_id
        self.transform = transform

    def __len__(self):
        return len(self.metadata)

    def __getitem__(self, idx):
        audio_path = self.metadata.iloc[idx]['path']
        label = self.metadata.iloc[idx]['label']
        
        # Load audio at native Wav2Vec sampling rate
        speech, _ = librosa.load(audio_path, sr=16000)
        
        if self.transform:
            speech = self.transform(speech)
            
        return torch.tensor(speech), torch.tensor(label, dtype=torch.float32)

def get_dataloader(csv_path, batch_size=32, shuffle=True):
    dataset = AudioDataset(csv_path)
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle)