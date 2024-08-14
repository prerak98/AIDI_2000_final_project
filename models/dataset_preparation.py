import librosa
import os
import pandas as pd
import numpy as np

def extract_features(file_name):
    audio_data, sample_rate = librosa.load(file_name, sr=None)
    mfccs = librosa.feature.mfcc(y=audio_data, sr=sample_rate, n_mfcc=13).mean(axis=1)
    chroma = librosa.feature.chroma_stft(y=audio_data, sr=sample_rate).mean(axis=1)
    spectral_contrast = librosa.feature.spectral_contrast(y=audio_data, sr=sample_rate).mean(axis=1)
    return mfccs, chroma, spectral_contrast

# Initialize an empty list to store data
feature_list = []

# Loop through all the audio files
for root, dirs, files in os.walk('../data/ravdess/'):
    for file in files:
        if file.endswith(".wav"):
            file_path = os.path.join(root, file)
            mfccs, chroma, spectral_contrast = extract_features(file_path)
            emotion_label = file.split('-')[2]
            
            feature_list.append({
                'mfcc': mfccs.tolist(),  # Save as list
                'chroma': chroma.tolist(),  # Save as list
                'spectral_contrast': spectral_contrast.tolist(),  # Save as list
                'emotion': emotion_label
            })

df = pd.DataFrame(feature_list)
df.to_json('../output/preprocessed_data.json', orient='records', lines=True)
print("Data saved to ../output/preprocessed_data.json")

