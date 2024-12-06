**TWITTER DATASET OVERVIEW**

**About the Dataset**  
This dataset contains Twitter messages and their emotional analysis. It has 11,000 rows and 5 columns. The dataset comprises vocal expressions conveyed through sentences that represent various scenarios using six emotions which are Sadness, Joy, Love Fear, Anger and Surprise.  
The dataset is in CSV format and can be used for text analysis and sentiment/emotion classification.  
The filenames adhere to a standardized format consisting of four distinct identifiers which are:  
**No:** The sequence of the data in rows. (No. 1- No. 11,000)

**Message**: Gives the content of the message (E.g. I feel awkward around rose)

**Source**: Indicated the origin of the data i.e. Twitter.

**Label**: A number categorizing a code for the emotion. (0,1,2,3,4,5)

**Emotion**: The emotion category of the message which are

* Sadness  
* Joy  
* Love  
* Fear  
* Anger  
* Surprise

**Data Preparation**  
**1.Data Manipulation and Analysis**

* Import pandas as pd-Provides tools for data manipulation and analysis, such as generating Data Frames and handling missing data.

* Import NumPy as np- Used for advanced mathematical operations on large numbers of data.

**2\. Data Visualization**

* Import matplotlib.pyplot as plt- Creating visualizations such as line graphs, histograms, etc.

* Import seaborn as sns- For creating statistical graphics, like heatmaps and pair plots.

**3\. Data Analysis and Summary Statistics**

* From collections import Counter- For counting the frequency of elements in an iterable (e.g., count word occurrences in a text).

**4\. Natural Language Processing (NLP)**

* From sklearn.feature\_extraction.text import TfidfVectorizer- Converts text data into numerical representations using Term Frequency-Inverse Document Frequency (TF-IDF), useful in NLP tasks for feature extraction.

* From nltk.tokenize import word\_tokenize-Splits sentences into individual words used for text preprocessing.

* From nltk.corpus import stopwords-Provides a collection of common words (like "the," "and") that can be removed to reduce noise in text data.

* From nltk.stem import PorterStemmer-Performs stemming, reducing words to their root forms to simplify text data.

* Import spacy-For advanced NLP tasks such as named entity recognition (NER), dependency parsing, and part-of-speech tagging.

**5\. Machine Learning and Preprocessing**

* From sklearn.model\_selection import train\_test\_split- Splits the dataset into training and testing subsets.

* From sklearn.model\_selection import GridSearchCV- Performs hyperparameter optimization by exhaustively searching over specified parameter values using cross-validation.

* From sklearn.preprocessing import StandardScaler, LabelEncoder- The StandardScaler  scales numerical features to have zero mean and unit variance, crucial for many ML models. And the LabelEncoder converts categorical labels into numeric form, often required for classification tasks.

* From sklearn.ensemble import RandomForestClassifier- Implements a powerful machine learning algorithm that builds multiple decision trees and combines their outputs for classification or regression tasks.

* from sklearn.linear\_model import LogisticRegression-Provides an effective algorithm for binary classification tasks.

**6\. Model Evaluation**

* From sklearn.metrics import classification\_report, accuracy score- The classification\_report: Displays precision, recall, F1-score, and support for each class in a classification task and accuracy score calculates the overall accuracy of the classification model.

**7\. Progress Tracking**

* fFom tqdm import tqdm- Adds a progress bar for iterations in loops, making it easier to monitor long-running processes.

**Folder Path**  
C:\\\\Users\\\\priya\\\\Documents\\\\Nehal\\\\GitHub\\\\msba265-finalstorage\\\\data\_storage\\\\Text\\\\twitter.csv

**Data Source**

The dataset is named **twitter.csv**, suggesting that the data consists of Twitter posts. The data was collected through tweets may have been gathered manually or exported from platforms/tools.

**Data Content**

From the CSV file:

* **Tweets (Messages)**: These are the core text messages collected from Twitter, stored in the Message column.

* **Metadata**: Includes fields like Source (indicating Twitter) and potential derived fields like Label (numeric sentiment) and Emotion (textual sentiment category).

**Dataframe** 

* The dataset was loaded using pandas.read\_csv:

i.e. data \= pd.read\_csv(r"C:\\\\Users\\\\priya\\\\Documents\\\\Nehal\\\\GitHub\\\\msba265-finalstorage\\\\data\_storage\\\\Text\\\\twitter.csv", encoding='latin1')

* The file **twitter.csv** was read from a specific local path, with latin1 encoding likely used to handle special characters.

**Content of the Dataframe**

The data frame **data** contains the following columns based on the earlier inspection of the dataset: No. , Message, source, label, emotion.

**1\. Handling Missing Values**

* **Missing Data Check**:

data.isnull().sum()

This step identifies missing values across all columns. If any columns have missing values (especially in No.), they are either filled, removed, or handled appropriately in later steps. This showed that No., Message, Source, Label and  Emotion had 1000 missing values each.

**2\. Text Preprocessing**

**Tokenization**: The code nltk.download('punkt') broke down text into individual words or tokens using NLTK's word\_tokenize.

**Stopwords Removal**: Words that are common (e.g., "the," "is") and do not contribute much meaning were removed using NLTK's stopword list.

**Stemming**: This step reduced words to their root form using NLTK's PorterStemmer. For example, "running" becomes "run".

**Lemmatization**: Used SpaCy for more advanced lemmatization.

**Text Normalization**: This involves making text uniform, such as converting to lowercase or removing special characters like punctuation, URLs, or numbers.  
**3\. Visualization**  
A **count plot** was used to visualize the relationship between the Emotion and Label columns. Label 1.0 had the highest count and label 2 had the lowest.  
A **histogram** was used visualize the distribution of message lengths with 30 bins. It also includes a KDE curve for a smoother representation of the data's underlying distribution.   
**Data Augmentation**  
**1\. Feature Engineering**

1. **Vectorization**: Converting text data into numerical features for machine learning models. One common method is **TF-IDF (Term Frequency-Inverse Document Frequency.** The parameter ngram\_range (1, 2\) was used to extract unigrams and bigrams from the text. The parameter stop\_words=’English’ was used to remove common stop words in English and max\_features limited the output to 2000\. This resulted in tthe **X\_vectorized.shape as (11000, 2000\) meaning** the dataset contains 11,000 samples  and each document is represented as a vector of 2,000 features (unigrams and bigrams) extracted and weighted using TF-IDFand **additional\_features.shape as (11000, 4\) meaning 11000** i.e. The same number of samples as X\_vectorized, ensuring compatibility and Each document having 4 additional features.

**Label Encoding**: The Emotion and Label columns to convert categorical labels into numerical values with the length of the data as 11,000 and 0 missing values in the label\_encoded column

2\. **Handling Imbalanced Data**

* **SMOTE (Synthetic Minority Oversampling Technique)**: If the dataset has imbalanced classes (e.g., certain emotions are underrepresented), SMOTE can be used to generate synthetic examples for minority classes.

**Sampling strategy for SMOTE**: 

* Class 0: Unsampled to 3,372 samples.  
* Classes 2, 3, 4, and 5: Each upsampled to 3,000 samples.

**Exploratory Data Analysis**  
A very detailed EDA was carried out the get important insights from the dataset. EDA such as missing data analysis and summary statistics, Visualizing Relationships, Distribution of Features

**Cross-Validation Check**  
\- The dataset was split into training (80%) and testing (20%) sets using Logistic regression and Random Forest Classification  
For Logistic Regression, Fitting was 5 folds for each of 5 candidates, totalling 25 fits  
The classification report summarizes the performance of a logistic regression model with an overall accuracy of 0.93. Here are the key points:

\- Class Performance:

  \- Class 0: Precision 0.93, Recall 0.89, F1-score 0.91, Support 675

  \- Class 1: Precision 0.94, Recall 0.85, F1-score 0.89, Support 674

  \- Class 2: Precision 0.90, Recall 0.98, F1-score 0.94, Support 600

  \- Class 3: Precision 0.93, Recall 0.96, F1-score 0.95, Support 600

  \- Class 4: Precision 0.92, Recall 0.96, F1-score 0.94, Support 600

  \- Class 5: Precision 0.95, Recall 0.95, F1-score 0.95, Support 600

\- Averages:

  \- Macro Average: Precision 0.93, Recall 0.93, F1-score 0.93

  \- Weighted Average: Precision 0.93, Recall 0.93, F1-score 0.93

Overall, the model performs well across all classes, with high precision, recall, and F1-scores.

Model Performance for Random Forest  
For Random Forest Fitting,5 folds for each of 27 candidates, totalling 135 fits

Overall Accuracy: The Random Forest model achieved an accuracy of 0.91.

Class Performance

\- Class 0:  Precision: 0.91, Recall: 0.84,  F1-score: 0.88,  Support: 675

\- Class 1:  Precision: 0.92, Recall: 0.94, F1-score: 0.93, Support: 600

\- Class 2:  Precision: 0.88, Recall: 0.94, F1-score: 0.91, Support: 600

\- Class 3: Precision: 0.93, Recall: 0.96, F1-score: 0.95, Support: 600

\- Class 4: Precision: 0.92, Recall: 0.96, F1-score: 0.94, Support: 600

\- Class 5: Precision: 0.98, Recall: 0.96, F1-score: 0.97, Support: 600

Averages

\- Macro Average: 

\- Precision: 0.9, Recall: 0.91, F1-score: 0.91

\- Weighted Average: Precision: 0.91, Recall: 0.91, F1-score: 0.91

The Random Forest model exhibits strong performance across all classes, with particularly high precision and recall for Class 5\. The overall metrics indicate that the model is effective in classifying the data, with consistent performance reflected in both macro and weighted averages.

