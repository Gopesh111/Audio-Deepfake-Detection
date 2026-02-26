import pandas as pd
from sklearn.model_selection import train_test_split

class SpeakerDisjointSplitter:
    """
    Ensures that no speaker's audio exists in both training and testing sets.
    Essential for validating model generalization across 21,140 unique speakers.
    """
    @staticmethod
    def split(metadata_csv: str, test_size: float = 0.2):
        df = pd.read_csv(metadata_csv)
        
        # Get unique speaker IDs
        unique_speakers = df['speaker_id'].unique()
        
        # Split at the speaker level, not the segment level
        train_speakers, test_speakers = train_test_split(
            unique_speakers, 
            test_size=test_size, 
            random_state=42
        )
        
        train_df = df[df['speaker_id'].isin(train_speakers)]
        test_df = df[df['speaker_id'].isin(test_speakers)]
        
        return train_df, test_df