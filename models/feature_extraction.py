import librosa
import os
import numpy as np
import pandas as pd

def extract_features(file_name):
    # Load the audio file
    audio_data, sample_rate = librosa.load(file_name, sr=None)
    
    # Extract MFCC features
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13).mean(axis=1)
    
    # Extract Chroma features
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate).mean(axis=1)
    
    # Extract Spectral Contrast features
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate).mean(axis=1)
    
    return mfccs, chroma, spectral_contrast

# Example usage
audio_dir = '../data/ravdess/Actor_01/'  # Example path
file_name = os.path.join(audio_dir, '03-01-01-01-01-01-01.wav')
mfccs, chroma, spectral_contrast = extract_features(file_name)

print(f"MFCCs: {mfccs}")
print(f"Chroma: {chroma}")
print(f"Spectral Contrast: {spectral_contrast}")
