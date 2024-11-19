import pandas as pd
import numpy as np
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from imblearn.over_sampling import SMOTE

df = pd.read_csv('/Users/roopasreesubramanyam/Desktop/MSBA 265/Project Data Analysis Data set/Dataset.csv')

print(df.head())
print(df.head())
print(df.columns)
df.columns = ['No', 'Text', 'emoji', 'social media', 'type', 'label'] + [f'Unnamed: {i}' for i in range(6, len(df.columns))]

print(df.columns)
df = df.drop(columns=['emoji'])

print(df.columns)
df = df[['No', 'Text', 'social media', 'type', 'label']]
df['Text'] = df['Text'].fillna('')
stop_words = set(stopwords.words('english'))

#Data Preprocessing
def preprocess_text(text):
    tokens = word_tokenize(text.lower()) 
    tokens = [word for word in tokens if word.isalnum() and word not in stop_words] 
    return ' '.join(tokens)

df['processed_text'] = df['Text'].apply(preprocess_text)

print(df[['Text', 'processed_text', 'label']].head())
df = df[pd.to_numeric(df['label'], errors='coerce').notnull()]

df['label'] = df['label'].astype(int)

print(df[['Text', 'processed_text', 'label']].head())
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 2))
X = tfidf_vectorizer.fit_transform(df['processed_text'])

print("TF-IDF matrix shape:", X.shape)
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score

#Testing and Training data
y = df['label']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)


print("Training set size:", X_train.shape, "Test set size:", X_test.shape)

model = LogisticRegression(max_iter=1000)  # Increased max_iter for convergence
model.fit(X_train, y_train)


y_pred = model.predict(X_test)
print("Accuracy:", accuracy_score(y_test, y_pred))
print("Classification Report:\n", classification_report(y_test, y_pred))
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score
from sklearn.feature_extraction.text import TfidfVectorizer
from imblearn.over_sampling import SMOTE
from sklearn.model_selection import GridSearchCV


X = tfidf_vectorizer.fit_transform(df['processed_text'])  
y = df['label']

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

smote = SMOTE(random_state=42)
X_train_res, y_train_res = smote.fit_resample(X_train, y_train)

log_reg = LogisticRegression(max_iter=1000)


param_grid = {
    'C': [0.1, 1, 10, 100],  # Regularization strengths
    'solver': ['liblinear', 'saga']  # Solvers for optimization
}

grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

best_model = grid_search.best_estimator_
y_pred = best_model.predict(X_test)
print("Improved Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Improved Logistic Regression Classification Report:\n", classification_report(y_test, y_pred))
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import RandomUnderSampler
from imblearn.pipeline import Pipeline

# Trying to improve the model to get the best fit 
smote = SMOTE(random_state=42)
under_sampler = RandomUnderSampler(random_state=42)

pipeline = Pipeline([('smote', smote), ('under_sampler', under_sampler)])

X_train_res, y_train_res = pipeline.fit_resample(X_train, y_train)

print("Resampled training set size:", X_train_res.shape, "Resampled labels size:", y_train_res.shape)
from sklearn.feature_extraction.text import TfidfVectorizer

tfidf_vectorizer = TfidfVectorizer(max_features=5000, ngram_range=(1, 3))
X = tfidf_vectorizer.fit_transform(df['processed_text'])

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

print("TF-IDF matrix shape:", X.shape)
from sklearn.linear_model import LogisticRegression

log_reg = LogisticRegression(max_iter=1000, class_weight='balanced')
log_reg.fit(X_train_res, y_train_res)

y_pred = log_reg.predict(X_test)
print("Improved Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Improved Logistic Regression Classification Report:\n", classification_report(y_test, y_pred))
from sklearn.model_selection import GridSearchCV

param_grid = {
    'C': [0.1, 1, 10, 100],
    'solver': ['liblinear', 'saga']  
}

grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

best_model = grid_search.best_estimator_

print("Best Hyperparameters:", grid_search.best_params_)

best_model.fit(X_train_res, y_train_res)

y_pred = best_model.predict(X_test)
print("Improved Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Improved Logistic Regression Classification Report:\n", classification_report(y_test, y_pred))
param_grid = {
    'C': [0.1, 1, 10],  
    'solver': ['liblinear']
}

grid_search = GridSearchCV(log_reg, param_grid, cv=5, scoring='accuracy', n_jobs=-1)
grid_search.fit(X_train_res, y_train_res)

best_model = grid_search.best_estimator_

best_model.fit(X_train_res, y_train_res)

y_pred = best_model.predict(X_test)
print("Improved Logistic Regression Accuracy:", accuracy_score(y_test, y_pred))
print("Improved Logistic Regression Classification Report:\n", classification_report(y_test, y_pred))
