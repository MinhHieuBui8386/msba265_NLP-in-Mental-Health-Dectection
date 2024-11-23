#Emojis analysis
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, confusion_matrix, classification_report

# Load the dataset from your CSV file
file_path = 'msba265-finalstorage\data_storage\emojis.csv'
df = pd.read_csv(file_path)

# Inspect the data to understand its structure
print(df.head())

# Step 1: Encode categorical columns (Label and Main Emotion)
label_encoder = LabelEncoder()
df['Label'] = label_encoder.fit_transform(df['Label'])
df['Main emotion'] = label_encoder.fit_transform(df['Main emotion'])

# Step 2: Feature Extraction (text -> numeric)
# Using CountVectorizer for emojis (treated as characters here)
vectorizer = CountVectorizer(analyzer='char')  # Using character-level analysis for emojis
X = vectorizer.fit_transform(df['Text'])  # Feature matrix

# Step 3: Define target variable and split the data
y = df['Label']  # or 'Main emotion', depending on what you want to predict

# Train-Test Split (80% training, 20% testing)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 4: Logistic Regression Model
model = LogisticRegression()
model.fit(X_train, y_train)

# Step 5: Prediction and Evaluation
y_pred = model.predict(X_test)

# Print Results
print(f'Accuracy: {accuracy_score(y_test, y_pred)}')
print(f'Confusion Matrix:\n{confusion_matrix(y_test, y_pred)}')
print(f'Classification Report:\n{classification_report(y_test, y_pred)}')
