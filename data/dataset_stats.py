import pandas as pd

def analyze_speaker_distribution(metadata_csv: str):
    """
    Analyzes the diversity of the dataset to verify 21,140 unique speakers.
    """
    df = pd.read_csv(metadata_csv)
    unique_speakers = df['speaker_id'].nunique()
    total_segments = len(df)
    
    print(f"Total Audio Segments: {total_segments}")
    print(f"Unique Speakers: {unique_speakers}")
    
    # Check for speaker-level balance
    speaker_counts = df.groupby('speaker_id').size()
    print(f"Average segments per speaker: {speaker_counts.mean():.2f}")
    
    return unique_speakers