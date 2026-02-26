import torch.nn as nn

class DeepfakeClassifier(nn.Module):
    """
    A Deep Learning model utilizing a MLP architecture to detect 
    synthetic audio outputs from Wav2Vec embeddings.
    """
    def __init__(self, input_dim: int = 768, hidden_dim: int = 256):
        super(DeepfakeClassifier, self).__init__()
        self.network = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.3),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Linear(hidden_dim // 2, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.network(x)