# Interpretation of EDA
## Emotion Distribution
- The distribution shows that the dataset is imbalanced, with neutral having the largest proportion (14.6%).
- Other emotions like angry, disgust, fear, happy, and sad are evenly distributed with 17.1%.
- Imbalance like this may affect model performance, as some emotions might dominate training results.

Next Steps: Might consider techniques to handle class imbalance, such as oversampling, undersampling, or using weighted loss during training.

## Audio Duration Distribution
- Most audio files have a duration between 2.0 to 3.0 seconds, with a peak around 2.5 seconds.
- Very few audio files have durations exceeding 4 seconds or shorter than 1.5 seconds.
- This indicates that the dataset is relatively consistent in terms of audio length.

## MFCC
- Description:

    - Transforming the audio signal from the time domain to the frequency domain (using Fourier transform).
    - Applying the Mel scale to mimic human hearing by emphasizing frequencies that are more perceptible to humans.
    - Extracting coefficients that summarize the spectral envelope.
    
- Insights:

    - Dominant Frequencies:

        - Red bands (high positive intensity) near the top suggest the presence of dominant frequencies or energy in the higher Mel bands.
        - Blue bands (low or negative values) near the bottom may represent silence, low energy, or the absence of certain frequencies.
    - Temporal Variation:

        - MFCC coefficients evolve over time, showing how the spectral characteristics of the audio signal change.
    - "Happy Emotion" Signature:

        - The distribution of red and blue areas can indicate features specific to "happy" emotions in audio, such as variations in pitch and energy.

## Waveplot
- Description:

    - This is a waveform plot representing the audio signal associated with the emotion "Sad".
    - The x-axis indicates time (seconds), while the y-axis shows amplitude, which represents the loudness or intensity of the sound at each point in time.
- Insights:

    - Peaks and troughs represent the variations in loudness.
    - The signal appears to be more concentrated in certain time ranges (e.g., around 1 second), which might suggest moments of higher intensity.
    - This kind of visualization is commonly used in speech emotion recognition to understand how emotions affect voice modulation.

## Spectrogram
- Description:

    - A spectrogram is a time-frequency representation of an audio signal.
    - The x-axis represents time (seconds), the y-axis represents frequency (Hz), and the color scale indicates intensity or energy levels in decibels (dB).
- Insights:

    - Darker areas represent lower intensity, while brighter areas (yellow-green) indicate higher energy at certain frequencies.
    - The spectrogram shows that certain frequency bands (e.g., below 1000 Hz and between 2000-3000 Hz) have more energy, which may be characteristic of the "sad" emotion.
    - Spectrograms help in understanding pitch, tone, and rhythm changes in speech or music that convey emotional information.

