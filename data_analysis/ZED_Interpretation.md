# Exploratory Data Analysis (EDA)

## Datasets:

- The Zaion Emotion Dataset (ZED) was developed by researchers Yingzhi Wang, Mirco Ravanelli, and Alya Yacoubi. 
- ZED is a speech emotion dataset curated from YouTube videos, capturing non-acted emotional expressions recorded in real-life settings. The videos feature a variety of emotional and interactive contexts such as comedy shows, victim interviews, sports commentary, etc.
- ZED comprises 180 audio files categorized into three most common emotions: happiness, sadness, or anger.
- This dataset is openly available to the research community for academic and scientific purposes. 

## Emotion Distribution

![alt text](image-1.png)

- The ZED emotions are fairly balanced, with sadness making up 36.7% of the dataset, happiness accounting for 34.4%, and the remaining portion being anger.
- The balanced distribution of emotions in the dataset is beneficial for building an emotion analysis model. It can ensure that the model learns patterns for each emotion equally thus improving its generalization to unseen data.

## Distribution of Audio Durations

![alt text](image.png)

- Most audio files have durations ranging from 1.0 to 15.0 seconds, with the majority peaking around 4.5 seconds. Only a small number of audio files exceed 10 seconds in duration.
- Long audio files (>10 seconds) could create variability, making the model's learning process less uniform, that could negatively impact the model's performance on both shorter and longer samples.

![alt text](image-3.png)

- The line inside the box represents the median (approximately 5 seconds) indicating that half of the audio files are shorter than this duration, while the other half are longer.
- Points beyond the whiskers (above ~10 seconds) are considered outliers. These represent audio files with durations significantly longer than the majority of the dataset.


## Distribution of Audio Durations

![alt text](image-4.png)

- The median duration is slightly longer for sad samples compared to angry and happy samples. Angry and happy have similar median durations, around 4–5 seconds.
- Sad and angry emotions have a few outliers, with durations exceeding 10 seconds.

## Correlation Heatmap for Extracted Features

![alt text](image-5.png)

- The heatmap shows how pairs of features are correlated with each other. 
    + Red areas: High positive correlation (close to 1.0) between two features, meaning that as one feature increases, the other tends to increase as well.
    + Blue areas: High negative correlation (close to -1.0), indicating an inverse relationship where one feature increases as the other decreases.
    + White areas: Near-zero correlation, showing little to no linear relationship between features.

## PCA (Principal Component Analysis):

![alt text](image-6.png)

- It visualizes data points representing three different emotional states: angry (blue), sad (orange), and happy (green). The data has been reduced to two dimensions - PCA Component 1 (x-axis) and PCA Component 2 (y-axis).
- The data points are fairly scattered across both components, with values ranging from approximately -6 to 8 on Component 1 and -6 to 8 on Component 2.
- There is significant overlap between the three emotional states, suggesting that the features used to distinguish between emotions are not perfectly separable.