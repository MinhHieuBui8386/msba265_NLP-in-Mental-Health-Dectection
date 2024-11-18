# Slang analysis 

import pandas as pd

# Load the dataset with a different encoding
file_path = r"C:\Users\nehal\Documents\GitHub\msba265-finalstorage\data_storage\slang.csv"
data = pd.read_csv(file_path, encoding='ISO-8859-1')  # Try using ISO-8859-1 or latin1 if this doesn't work


# Display the first few rows to understand the dataset structure
print(data.head())

# Data Cleaning
# Check for missing values
print("Missing values per column:\n", data.isnull().sum())

# Drop rows with missing values or handle them accordingly
data_cleaned = data.dropna()  # Drop rows with missing values, or use fillna() to fill them

# Standardize column names (remove extra spaces or special characters)
data_cleaned.columns = data_cleaned.columns.str.strip().str.lower().str.replace(" ", "_")

# Check for duplicates and drop if necessary
print(f"Duplicates before: {data_cleaned.duplicated().sum()}")
data_cleaned = data_cleaned.drop_duplicates()
print(f"Duplicates after: {data_cleaned.duplicated().sum()}")

# Display the cleaned dataset
print(data_cleaned.head())
