import pandas as pd
import matplotlib.pyplot as plt

# load emotion data
fer2013 = pd.read_csv('msba265-finalstorage/data_storage/fer2013.csv')
label_to_emotion = {
    0: "Angry",
    1: "Disgust",
    2: "Fear",
    3: "Happy",
    4: "Sad",
    5: "Surprise",
    6: "Neutral"
}

# convert emotion label to emotion name 
fer2013['emotion_name'] = fer2013['emotion'].map(label_to_emotion)

# count emotion 
emotion_counts = fer2013['emotion_name'].value_counts()

# plot a bar chart 
plt.figure(figsize=(8, 6))
emotion_counts.plot(kind='bar', color='skyblue', alpha=0.8)
plt.title('Emotion Distribution (FER2013 dataset)', fontsize=16)
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()

#load facial emotion data
fe = pd.read_csv('msba265-finalstorage/data_storage/facial_adjusted.csv')
# count emoion
emotion_counts = fe['emotion'].value_counts()

# plot a bar chart 
plt.figure(figsize=(8, 6))
emotion_counts.plot(kind='bar', color='skyblue', alpha=0.8)
plt.title('Emotion Distribution (Facial Emotion dataset)', fontsize=16)
plt.xlabel('Emotion', fontsize=12)
plt.ylabel('Count', fontsize=12)
plt.xticks(rotation=45)
plt.grid(axis='y', linestyle='--', alpha=0.7)
plt.show()