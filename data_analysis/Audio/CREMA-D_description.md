# CREMA-D AUDIO ANALYSIS: AN OVERVIEW

## About the Dataset
**CREMA-D** (Crowd-sourced Emotional Multimodal Actors Dataset) contains 7,442 audio clips recorded by 91 actors and actresses with diverse age and ethnicity. The dataset consists of vocal expressions in sentences representing six emotions (Anger, Disgust, Fear, Happy, Neutral, and Sad) across four intensity levels (Low, Medium, High, and Unspecified).  

All audio files in the CREMA-D dataset are in `.WAV` format, commonly used for computational audio processing. The filenames follow a standardized structure with four identifiers:  
1. **Actor ID**: A 4-digit number identifying the actor.  
2. **Sentences**: Actors spoke from a selection of 12 sentences:
   - *It's eleven o'clock* (IEO).  
   - *That is exactly what happened* (TIE).  
   - *I'm on my way to the meeting* (IOM).  
   - *I wonder what this is about* (IWW).  
   - *The airplane is almost full* (TAI).  
   - *Maybe tomorrow it will be cold* (MTI).  
   - *I would like a new alarm clock* (IWL).  
   - *I think I have a doctor's appointment* (ITH).  
   - *Don't forget a jacket* (DFA).  
   - *I think I've seen this before* (ITS).  
   - *The surface is slick* (TSI).  
   - *We'll stop in a couple of minutes* (WSI).  
3. **Emotions**: Presented in 6 different emotions:  
   - Anger (ANG)  
   - Disgust (DIS)  
   - Fear (FEA)  
   - Happy/Joy (HAP)  
   - Neutral (NEU)  
   - Sad (SAD)  
4. **Intensity Level**: Expressed at 4 different levels:  
   - Low (LO)  
   - Medium (MD)  
   - High (HI)  
   - Unspecified (XX)  

Each identifier is separated by underscores (`_`) in the filename. *(Cao et al., 2014)*  

---

## Data Preparation
### Import Necessary Libraries  
Each library serves a specific purpose, and their combination enables us to: (1) process data, (2) analyze audio signals, (3) visualize data and audio features, and (4) perform machine learning tasks. The following libraries were used:  
- **`os`**: This library is used to navigate directories, manage file paths, and load audio files.  
- **`numpy`**: This library is used for numerical computations in Python.  
- **`pandas`**: This library is used for data manipulation and analysis. It’s also used for organizing, cleaning, and preprocessing metadata or features extracted from audio files.  
- **`librosa` and `librosa.display`**: These are specialized libraries for audio analysis.
    + The library **`librosa`** provide tools for audio processing, such as loading audio files, extracting features (e.g., spectrograms, MFCCs), and computing transforms
    + The library **`librosa.display`** helps visualize audio features (e.g., waveforms, spectrograms) for better understanding and analysis.
- **`seaborn`**: This library is used for statistical data visualizations (e.g., emotion distributions, corelations)
-  **`matplotlib.pyplot`**: This library is often used in conjunction with “seaborn” for more advanced visualizations (e.g., waveforms, spectrograms, or extracted audio feature trends).  
- **`sklearn.model_selection`** (specifically “train_test_split”): This module is part of the **`scikit-learn`** library, commonly used for machine learning tasks such as cross-validation and hyperparameter tuning. It is also used to split the dataset into training and testing subsets for model evaluation.  

###  Define the Folder Path  
The folder containing all audio files is located at:  
`data_storage/Audio/CREAMA-D/AudioWAV`. 

This path will be used multiple times during the EDA and modeling process.

###  Data Collection  
- All audio file paths are gathered from the specified directory then be stored in an array called `audio_path`.

- Another array, `audio_emotion`, is created to store the corresponding emotion labels extracted from the filenames. 

- These labels are then mapped to a standardized format (e.g., converting “ANG” to “angry”) to ensure consistency in data labeling.  
  

### Create a DataFrame  
A dataframe named `dataset` was created with two columns:  
- **`Path`**: Contains audio file paths.  
- **`Emotion`**: Contains the extracted emotion labels corresponding to each path.  

---

## Exploratory Data Analysis (EDA)

Extensive EDA was conducted to derive meaningful insights from the dataset.

### Dataset Overview
1. **Dataset structure**:  
   - The function dataset.info() is used to summarize the dataset structure. 
   - The dataset contains **7,442 entries** and **2 columns** (`Path` and `Emotions`). Both columns contain non-null string data.  

2. **Emotion Distribution**:  
   - The function `dataset['Emotions'].value_counts()` is used to count occurrences of each emotion label.
   - The result shows an equal distribution among most emotions (1,271 instances for Angry, Disgust, Fear, Happy, and Sad).  
   - Neutral has fewer samples (1,087), making up 14.6% of the dataset.  

### Visualization of Emotion Distribution
- We use a histogram to visualize the distribution of emotions in the dataset, with a unique color scheme for each emotion type. The plot highlights both the count and percentage of each emotion.  
- Counts range from approximately 1,000 to 1,300. Most emotions (Angry, Disgust, Fear, Happy, Sad) are evenly distributed, each comprising 17.1% of the dataset (around 1,271 samples), while Neutral has fewer samples (1,087), making up the smallest proportion at 14.6%, indicating a slight imbalance in the dataset.  

### Distribution of Audio Durations
- Most audio files have durations between **2.0 to 3.0 seconds**, peaking at 2.5 seconds.  
- There are few audio files with durations shorter than 1.5 seconds or longer than 4.0 seconds.  
- The distribution is slightly right-skewed, with a long tail extending towards longer durations. It means that the dataset has a consistent range of audio lengths.

---

## Data Augmentation
The main purpose of this step is to enhance the dataset's diversity to prevent overfitting and improve the generalization of speech emotion recognition models. Many features such as pitch, duration, loudness, voice quality, etc. contribute to the transmission of emotional content in voice. Thus, we define 4 functions to ensure the model performs well under real-world conditions: 

1. **`awgn(data)`**: This function adds Additive White Gaussian Noise (AWGN) to audio data.

- The sample audio data is too clean, while real-world speech usually includes background noise. Thus, adding noise could increase the diversity of training data, improving the model's ability to generalize to unseen data. 
- It also helps the model distinguish between noise and key features (e.g., emotions in speech), ensuring robust performance in real-world environments

2. **`pitch(data, sr=44100, pitch_factor=0.7)`**: This function shifts the pitch of the audio data. 
- In real-world scenarios, speech naturally varies in pitch due to factors like speaker characteristics (e.g., gender, age) or emotional states. 
- Pitch shifting helps the model adapt to different vocal ranges, such as low-pitched voices (e.g., adult males) or high-pitched voices (e.g., children, females). Thus, it could increase the dataset's diversity and reduce the risk of overfitting to the original data.

3. **`stretch(data, rate=0.8)`**: This function stretches or compresses the audio duration while preserving the pitch.
- The speaking rate usually varies depending on the speaker’s mood, age or conversational context. For example, excited or stressed speakers may speak faster, while calm or hesitant speakers may speak slower. This factor should be kept subtle, so we chose the rate of 0.8x slower than the original duration. 
- Pitch also reflects speaker-specific traits such as gender, age, vocal tone etc. Thus, maintaining these characteristics ensures the model remains consistent and performs effectively across diverse real-world scenarios.
  
4. **`time_shift(data, sr=44100, shift_limit=1)`**: This function performs a time shift by rolling the audio data array.  
- In real-world scenarios, speech signals are often misaligned due to factors like noise or silence at the beginning or end of the recording.
- Time shifting helps train the model to focus on the audio content rather than relying on specific temporal alignments, improving its robustness to such variations.


---

## EDA with Augmented MFCCs
- In this step, we will perform an Exploratory Data Analysis (EDA) on augmented Mel Frequency Cepstral Coefficients (MFCCs) for the audio dataset labeled by emotions. For each unique emotion in the dataset, it selects a sample audio file, loads it using librosa library, and applies several audio augmentation techniques including the original audio, additive white Gaussian noise (AWGN) combined with time shifting, pitch shifting combined with time shifting, and time stretching. MFCCs are then calculated for each augmentation using function librosa.feature.mfcc, which are visualized as spectrograms using function **`librosa.display.specshow`**.  

- These visualizations allow for comparative analysis of how augmentations alter the MFCC representation for different emotions. The x-axis represents time and the y-axis represents the MFCC coefficients. The color scale indicates the intensity or magnitude of the MFCCs, with warmer colors (like red) representing higher values and cooler colors (like blue) representing lower values.  

---

## Outlier Detection
- In this step, we identify audio files with unusually long durations, which are those exceeding the 99th percentile of the “Duration” column in the dataset.  
- The result shows that **70 files** with duration above 4.04 seconds were identified as outliers. 

---

## Cross-Validation Check
- The dataset was split into training (80%) and testing (20%) sets. 

- The result shows that the **Training set** consists of 5,953 samples, with 869 labeled as neutral and 1,017 samples for each of the other emotions. Similarly, the **Test set** contains 1,489 samples, with 218 labeled as neutral and 254 for each of the other emotions

---

## References
Cao, H., Cooper, D. G., Keutmann, M. K., Gur, R. C., Nenkova, A., & Verma, R. (2014). *CREMA-D: Crowd-sourced Emotional Multimodal Actors Dataset.* IEEE Transactions on Affective Computing, 5(4), 377–390. [https://doi.org/10.1109/TAFFC.2014.2336244](https://doi.org/10.1109/TAFFC.2014.2336244)
