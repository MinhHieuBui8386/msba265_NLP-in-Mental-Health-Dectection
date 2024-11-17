import os
import numpy as np
import pandas as pd
import librosa
import librosa.display
import seaborn as sns
import matplotlib.pyplot as plt

# Define paths
path = 'msba265-finalstorage/data_storage/CREAMA-D/AudioWAV/'
if not os.path.exists(path):
    raise FileNotFoundError(f"Path does not exist: {path}")

# Collect audio file paths and emotions
audio_path = []
audio_emotion = []
print("Collecting audio file paths and extracting emotions...")
for audio in os.listdir(path):
    full_path = os.path.join(path, audio)
    if os.path.isfile(full_path):
        audio_path.append(full_path)
        emotion = audio.split('_')[2]
        emotion_map = {
            "SAD": "sad",
            "ANG": "angry",
            "DIS": "disgust",
            "NEU": "neutral",
            "HAP": "happy",
            "FEA": "fear"
        }
        audio_emotion.append(emotion_map.get(emotion, "unknown"))

# Create a dataset
emotion_dataset = pd.DataFrame(audio_emotion, columns=['Emotions'])
audio_path_dataset = pd.DataFrame(audio_path, columns=['Path'])
dataset = pd.concat([audio_path_dataset, emotion_dataset], axis=1)
print(f"Dataset created with {len(dataset)} entries.")
print(dataset.head())

# Visualization of Emotion Distribution
plt.figure(figsize=(6, 6), dpi=80)
sns.histplot(dataset.Emotions, color='#F19C0E')
plt.title("Emotion Count", size=16)
plt.xlabel('Emotions', size=12)
plt.ylabel('Count', size=12)
plt.show()

# Define function for waveplot and spectrogram visualization
def plot_wave_and_spectrogram(file_path, emotion):
    try:
        data, sampling_rate = librosa.load(file_path, sr=16000)  # Consistent sampling rate
        plt.figure(figsize=(10, 6))
        plt.title(f"Waveplot for {emotion.capitalize()} Emotion", size=16)
        librosa.display.waveshow(data, sr=sampling_rate)
        plt.show()

        plt.figure(figsize=(10, 4))
        plt.title(f"Spectrogram for {emotion.capitalize()} Emotion", size=16)
        D = librosa.stft(data)
        S_db = librosa.amplitude_to_db(np.abs(D), ref=np.max)
        librosa.display.specshow(S_db, sr=sampling_rate, x_axis='time', y_axis='hz', cmap='viridis')
        plt.colorbar(format='%+2.0f dB')
        plt.show()
    except Exception as e:
        print(f"Error processing file {file_path}: {e}")

# Select a sample file for the 'sad' emotion
sad_files = dataset[dataset['Emotions'] == 'sad']['Path']
if not sad_files.empty:
    sample_file = sad_files.values[0]
    print(f"Processing file: {sample_file}")
    plot_wave_and_spectrogram(sample_file, "sad")
else:
    print("No files with 'sad' emotion found.")

# Feature extraction function
def extract_features(dataset, target_length=5000):
    X, Y = [], []
    print("Extracting features from audio files...")
    for path, emotion in zip(dataset['Path'], dataset['Emotions']):
        try:
            # Load audio file with consistent sampling rate
            value, sample_rate = librosa.load(path, sr=16000)

            # Add noise to audio
            noise_amp = 0.035 * np.random.uniform() * np.amax(value)
            value = value + noise_amp * np.random.normal(size=value.shape[0])

            # Extract MFCC features
            mfcc = librosa.feature.mfcc(y=value, sr=sample_rate, n_mfcc=13, n_fft=200, hop_length=512)
            mfcc = mfcc.T.flatten()

            # Extract Mel Spectrogram features
            mel = librosa.feature.melspectrogram(y=value, sr=sample_rate, hop_length=256, n_fft=512, n_mels=64)
            mel = librosa.power_to_db(mel ** 2).T.flatten()

            # Combine features
            features = np.hstack((mfcc, mel))

            # Pad or truncate features to ensure consistent length
            if len(features) > target_length:
                features = features[:target_length]
            else:
                features = np.pad(features, (0, max(0, target_length - len(features))), 'constant')

            X.append(features)
            Y.append(emotion)

        except Exception as e:
            print(f"Error processing file {path}: {e}")

    return np.array(X), np.array(Y)

# Extract features from a subset of the dataset (demo with first 50 samples)
X, Y = extract_features(dataset.head(50))
if X.size > 0:
    print(f"Features extracted: {X.shape}")
    extracted_audio_df = pd.DataFrame(X)
    extracted_audio_df["Emotion"] = Y
    print(extracted_audio_df.head())
else:
    print("No features extracted. Check your dataset or file paths.")
