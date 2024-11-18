import matplotlib
matplotlib.use('Agg') 
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

# Load the dataset from the specified file path
file_path = r"C:\Users\nehal\Documents\GitHub\msba265-finalstorage\data_storage\emojis.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Adjust encoding if needed

# Display the first few rows of the dataset
print(data.head())

# Check for missing values in the dataset
print("Missing values per column:")
print(data.isnull().sum())

# Count the occurrences of 'Main emotion' and 'Second emotion'
print("\nMain emotion distribution:")
print(data['Main emotion'].value_counts())

print("\nSecond emotion distribution:")
print(data['Second emotion'].value_counts())

# Save the distribution of main emotions as a PNG file
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Main emotion', order=data['Main emotion'].value_counts().index)
plt.title('Distribution of Main Emotions')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()  # To avoid cutting off labels
plt.savefig("main_emotion_distribution.png")  # Save as PNG file

# Save the distribution of second emotions as a PNG file
plt.figure(figsize=(10, 6))
sns.countplot(data=data, x='Second emotion', order=data['Second emotion'].value_counts().index)
plt.title('Distribution of Second Emotions')
plt.xlabel('Emotion')
plt.ylabel('Count')
plt.xticks(rotation=45)
plt.tight_layout()  # To avoid cutting off labels
plt.savefig("second_emotion_distribution.png")  # Save as PNG file

# Optionally, inspect the unicode column and see how emojis map to their codes
print("\nUnicode values:")
print(data[['Text', 'unicode']].head())
