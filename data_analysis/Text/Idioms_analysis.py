# Import necessary libraries
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.preprocessing import LabelEncoder, MinMaxScaler



# Load the dataset from the provided file path
data = pd.read_csv(r"C:\Users\nehal\Documents\GitHub\msba265-finalstorage\data_storage\idioms.csv")

# Data Cleaning
print("Missing values per column:\n", data.isnull().sum())
data_cleaned = data.dropna()  # Drop rows with missing values

# Check and remove duplicates based on the 'Text' column
print(f"Duplicates before: {data_cleaned.duplicated(subset=['Text']).sum()}")
data_cleaned = data_cleaned.drop_duplicates(subset=['Text'])
print(f"Duplicates after: {data_cleaned.duplicated(subset=['Text']).sum()}")

# Standardizing column names
data_cleaned.columns = data_cleaned.columns.str.strip().str.lower().str.replace(" ", "_")

# Display the cleaned data
print(data_cleaned.head())

# Exploratory Data Analysis (EDA)
# Distribution of labels
sns.countplot(data=data_cleaned, x='label')
plt.title("Label Distribution")
plt.savefig('label_distributions.png')  # Save the plot as an image file


# Distribution of emotions
sns.countplot(data=data_cleaned, y='main_emotion', order=data_cleaned['main_emotion'].value_counts().index)
plt.title("Main Emotion Distribution")
plt.savefig('main_emotion_distribution.png')  # Save the plot as an image file


# Adding text length as a new feature
data_cleaned['text_length'] = data_cleaned['text'].apply(len)

# Distribution of text lengths
sns.histplot(data=data_cleaned, x='text_length', bins=30, kde=True)
plt.title("Text Length Distribution")
plt.savefig('text_length_distribution.png')  # Save the plot as an image file


# Data Transformation
# Encoding 'Type' and 'Main emotion' using LabelEncoder
encoder = LabelEncoder()
data_cleaned['type_encoded'] = encoder.fit_transform(data_cleaned['type'])
data_cleaned['main_emotion_encoded'] = encoder.fit_transform(data_cleaned['main_emotion'])

# Vectorizing the 'Text' column using CountVectorizer
vectorizer = CountVectorizer(stop_words='english', max_features=500)
text_vectors = vectorizer.fit_transform(data_cleaned['text']).toarray()

# Adding vectorized text as a new dataframe
vectorized_df = pd.DataFrame(text_vectors, columns=vectorizer.get_feature_names_out())
data_transformed = pd.concat([data_cleaned.reset_index(drop=True), vectorized_df], axis=1)

# Normalize 'text_length' using MinMaxScaler
scaler = MinMaxScaler()
data_transformed['text_length_normalized'] = scaler.fit_transform(data_transformed[['text_length']])

# Drop original 'Text' column if no longer needed
data_transformed = data_transformed.drop(columns=['text'])

# Preview the transformed dataset
print(data_transformed.head())

# Distribution Analysis
# Shape of the transformed data
print("Shape of transformed data:", data_transformed.shape)

# Label distribution
label_distribution = data_transformed['label'].value_counts(normalize=True)
print("Label Distribution:\n", label_distribution)

# Emotion distribution
emotion_distribution = data_transformed['main_emotion_encoded'].value_counts(normalize=True)
print("Main Emotion Distribution:\n", emotion_distribution)

# Text length analysis
print("Text Length Stats:\n", data_transformed['text_length_normalized'].describe())
