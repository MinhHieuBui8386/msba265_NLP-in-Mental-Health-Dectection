# Twitter Text Analysis

# Set the Matplotlib backend to 'Agg' for non-GUI plotting
import matplotlib
matplotlib.use('Agg')

# Import necessary libraries
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler

# Load the dataset from the provided file path
data = pd.read_csv(r"C:\Users\nehal\Documents\GitHub\msba265-finalstorage\data_storage\twitter.csv")

# Display the first few rows
print(data.head())

# Data Cleaning
print("Missing values per column:\n", data.isnull().sum())
data_cleaned = data.dropna()  # Drop rows with missing values

# Check and remove duplicates based on the 'Message' column
print(f"Duplicates before: {data_cleaned.duplicated(subset=['Message']).sum()}")
data_cleaned = data_cleaned.drop_duplicates(subset=['Message'])
print(f"Duplicates after: {data_cleaned.duplicated(subset=['Message']).sum()}")

# Standardizing column names
data_cleaned.columns = data_cleaned.columns.str.strip().str.lower().str.replace(" ", "_")

# Display the cleaned data
print(data_cleaned.head())

# Exploratory Data Analysis (EDA)
# Distribution of labels
sns.countplot(data=data_cleaned, x='label')
plt.title("Label Distribution")
plt.savefig('label_distribution.png')  # Save the figure instead of showing it

# Distribution of emotions
sns.countplot(data=data_cleaned, y='emotion', order=data_cleaned['emotion'].value_counts().index)
plt.title("Emotion Distribution")
plt.savefig('emotion_distribution.png')  # Save the figure instead of showing it

# Adding message length as a new feature
data_cleaned['message_length'] = data_cleaned['message'].apply(len)

# Distribution of message lengths
sns.histplot(data=data_cleaned, x='message_length', bins=30, kde=True)
plt.title("Message Length Distribution")
plt.savefig('message_length_distribution.png')  # Save the figure instead of showing it

# Data Transformation
# Encoding 'Source' and 'Emotion' using LabelEncoder
encoder = LabelEncoder()
data_cleaned['source_encoded'] = encoder.fit_transform(data_cleaned['source'])
data_cleaned['emotion_encoded'] = encoder.fit_transform(data_cleaned['emotion'])

# Vectorizing the 'Message' column using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=500)
message_vectors = vectorizer.fit_transform(data_cleaned['message']).toarray()

# Adding vectorized text as a new dataframe
vectorized_df = pd.DataFrame(message_vectors, columns=vectorizer.get_feature_names_out())
data_transformed = pd.concat([data_cleaned.reset_index(drop=True), vectorized_df], axis=1)

# Normalize 'message_length' using MinMaxScaler
scaler = MinMaxScaler()
data_transformed['message_length_normalized'] = scaler.fit_transform(data_transformed[['message_length']])

# Drop original 'Message' column if no longer needed
data_transformed = data_transformed.drop(columns=['message'])

# Preview the transformed dataset
print(data_transformed.head())

# Distribution Analysis
# Shape of the transformed data
print("Shape of transformed data:", data_transformed.shape)

# Label distribution
label_distribution = data_transformed['label'].value_counts(normalize=True)
print("Label Distribution:\n", label_distribution)

# Emotion distribution
emotion_distribution = data_transformed['emotion_encoded'].value_counts(normalize=True)
print("Emotion Distribution:\n", emotion_distribution)

# Message length analysis
print("Message Length Stats:\n", data_transformed['message_length_normalized'].describe())
