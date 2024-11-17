# CREAMA-D Analysis
import pandas as pd

file_path = 'msba265-finalstorage/data_storage/CREAMA-D.csv'
data = pd.read_csv(file_path)
data_info = data.info()
data_head = data.head()

print(data_info, data_head)

# Clean data
cleaned_data = data.dropna(axis=1, how='all')
cleaned_data.columns = ["Clip Number", "Clip Name", "Response Level", "Displayed Emotion"]
missing_values = cleaned_data.isnull().sum()
data_types = cleaned_data.dtypes
cleaned_data = cleaned_data.drop_duplicates()

cleaned_data_info = cleaned_data.info()
cleaned_data_head = cleaned_data.head()

print(missing_values, data_types, cleaned_data_info, cleaned_data_head)


# Preprocessing data
cleaned_data["Response Level"] = pd.to_numeric(cleaned_data["Response Level"], errors='coerce')
response_level_stats = cleaned_data["Response Level"].describe()
emotion_distribution = cleaned_data["Displayed Emotion"].value_counts()
emotion_response_avg = cleaned_data.groupby("Displayed Emotion")["Response Level"].mean()

print(response_level_stats)
print(emotion_distribution)
print(emotion_response_avg)

import matplotlib.pyplot as plt
import seaborn as sns

# Remove the ambiguous "Displayed Emotion" entry
cleaned_data = cleaned_data[cleaned_data["Displayed Emotion"] != "Displayed Emotion"]

# Visualize distributions and box plots
plt.figure(figsize=(12, 6))
# Distribution of Response Levels
sns.histplot(cleaned_data["Response Level"], bins=30, kde=True, color='skyblue')
plt.title("Distribution of Response Levels")
plt.xlabel("Response Level")
plt.ylabel("Frequency")
plt.show()

# Distribution of Displayed Emotions
plt.figure(figsize=(12, 6))
sns.countplot(data=cleaned_data, x="Displayed Emotion", palette="viridis")
plt.title("Distribution of Displayed Emotions")
plt.xlabel("Emotion")
plt.ylabel("Count")
plt.show()

# Box plot of Response Levels across Emotions
plt.figure(figsize=(12, 6))
sns.boxplot(data=cleaned_data, x="Displayed Emotion", y="Response Level", palette="pastel")
plt.title("Box Plot of Response Levels by Displayed Emotion")
plt.xlabel("Emotion")
plt.ylabel("Response Level")
plt.show()

# Outliers in Response Level
outliers = cleaned_data[
    (cleaned_data["Response Level"] < cleaned_data["Response Level"].quantile(0.05)) |
    (cleaned_data["Response Level"] > cleaned_data["Response Level"].quantile(0.95))
]

len(outliers), outliers.head()
