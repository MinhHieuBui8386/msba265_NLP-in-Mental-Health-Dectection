import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import json
import librosa
import librosa.display
import IPython.display as ipd
import os

# Google Drive Integration
from google.colab import drive
drive.mount('/content/drive')

# Load the JSON file 
ZED = pd.read_json('/content/drive/My Drive/ZED/ZED.json')
ZED.head()

# Data Transformation

audio_paths = []
durations = []
audio_emotions = []

for key, value in ZED.items():
    audio_path = value['wav'].replace("datafolder/", "")
    duration = value['duration']
    
    for emotion_data in value['emotion']:
        audio_paths.append(audio_path)
        audio_emotions.append(emotion_data['emo'])
        durations.append(duration) 

df = pd.DataFrame({
    'Audio_path': audio_paths,
    'Duration': durations,
    'Emotion': audio_emotions})

print(df)

# Data Summary

# Total number of audio files
total_files = len(durations)
print(f"Total number of audio files: {total_files}")

# Calculate average, min, and max durations
average_duration = np.mean(durations)
min_duration = np.min(durations)
max_duration = np.max(durations)
print(f"Average duration: {average_duration:.2f} seconds")
print(f"Minimum duration: {min_duration:.2f} seconds")
print(f"Maximum duration: {max_duration:.2f} seconds")

# Exploratory Analysis

#Create a histogram of emotion distribution
plt.figure(figsize=(5, 5))
emotion_counts = df['emotion'].value_counts()
bar_plot = emotion_counts.plot(kind='bar', color='salmon')
for index, value in enumerate(emotion_counts):
    plt.text(index, value, str(value), ha='center', va='bottom')
plt.title('Emotion Distribution in ZED Dataset')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.show()


# Feature Extraction from Audio
path = '/content/drive/My Drive/ZED'
def feature_extraction(path):
    try:
        # Load the audio file 
        X, sample_rate = librosa.load(path, sr=None, res_type='kaiser_fast')

        # Extract MFCC features from the audio
        mfcc = np.mean(librosa.feature.mfcc(y=X, sr=sample_rate, n_mfcc=50).T, axis = 0)

        return mfcc
    except Exception as e:
        print(f"Error processing {path}: {e}")
        return None

features = {}
# Iterate through all files in the directory
for audio in os.listdir(path):
    audio_path = os.path.join(path, audio)
    if os.path.isfile(audio_path):
        mfcc_features = feature_extraction(audio_path)
        if mfcc_features is not None: 
            features[audio] = mfcc_features

Print the extracted features
print(f"Extracted features for {len(features)} audio files.")