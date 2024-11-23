# Interpretation of EDA
## fer2013_Facial Emotion
- The dataset contains 48x48 pixel grayscale images of faces. These images have been preprocessed to ensure that the faces are approximately centered and occupy a similar amount of space within each frame.
- Categorize each face based on the emotion shown in the facial expression into one of seven categories (0=Angry, 1=Disgust, 2=Fear, 3=Happy, 4=Sad, 5=Surprise, 6=Neutral).
## EDA
1. Basic Statistics for Numeric Columns
- Compute summary statistics for the numeric columns in the two datasets.
- This function generates descriptive statistics that summarize the central tendency, dispersion, and shape of a dataset’s distribution.
2. Check Class Distribution
- A count for each unique class in the emotion column (e.g., how many times "Happy", "Sad", etc., appear in the dataset).
3. Map and Count Classes in fer2013
- Maps numeric labels in the emotion column of the fer2013 dataset to their corresponding emotion names.
- Counts the occurrences of each mapped class.
## Image file conversion
- Normalize Pixel Values in the fer2013 Dataset
- Converts pixel data stored as strings into numerical arrays and then normalizes the pixel values to a range of 0 to 1.
- Pixel values typically range from 0 to 255.
- Dividing each pixel value by 255 scales the data to the range [0, 1].
## Emotion Label Preparation
- Converts the emotion column into a one-hot encoded format.
- Converts categorical data (e.g., emotions like "Happy", "Sad", "Neutral") into a binary matrix.
- Each emotion becomes its own column, with a value of 1 if the row corresponds to that emotion, and 0 otherwise.


