# Speech Emotion Recognition

## Importing Libraries
Each Python library serves a specific purpose. The following libraries were used:

### General Libraries
1. **IPython**: Used for interactively running Python code.
2. **os**: Used to navigate directories, manage file paths, and load audio files. 
3. **pathlib.Path**: Ensures that all file paths work consistently across different operating systems.
4. **random**: Used for generating random numbers or selections.
5. **PIL.Image**: Handles image loading and processing.
6. **scipy.io**: Provides functions for reading and writing data in various formats (e.g., reading WAV files).

### Data Analysis and Preprocessing
1. **librosa**: Provides tools for audio processing, such as feature extraction (e.g., MFCCs).
2. **numpy**: Used for numerical computations in Python.
3. **pandas**: Used for data manipulation and analysis, including organizing, cleaning, and preprocessing metadata or features extracted from audio files. 
4. **Seaborn**: Provides advanced statistical data visualizations.
5. **sklearn.model_selection**: Includes `train_test_split` to divide datasets into training and testing subsets for model evaluation.
6. **sklearn.preprocessing**: Includes `StandardScaler` for feature scaling and `LabelEncoder` for categorical label conversion.
7. **sklearn.metrics**: Includes `accuracy_score`, `classification_report`, and `confusion_matrix` for evaluating model performance.

### Visualization
1. **matplotlib.pyplot**: Used for plotting and visualizing data (e.g., histograms, waveforms, etc.).
2. **%matplotlib inline**: Ensures that plots are displayed directly in Jupyter Notebook.

### Modeling
1. **tensorflow** and **keras**: Used for building and training deep learning models. Key components include:
   - **Sequential**: Implements simple layer-by-layer architectures without multiple inputs, multiple outputs, or layer branches.
   - **Layers**: Modules like `Dense`, `Dropout`, `Conv1D`, `Conv2D`, and others are used for building neural networks.
   - **Optimizers**: Algorithms such as Adam or SGD are used for training models.
   - **Callbacks**: These functions will be executed at specific stages of the training process:
     - **EarlyStopping**: This callback stops training the model when performance stops improving. It helps prevent overfitting and saves time.
     - **ModelCheckpoint**: This callback saves the model at specified intervals or based on certain conditions during training to track the best version.
     - **LearningRateScheduler**: This callback adjusts the learning rate during training for optimization, improving model performance.
2. **keras_self_attention.SeqSelfAttention**: Implements an attention mechanism that processes sequential data by capturing contextual relationships across timestamps.

## Utility Functions

- **`find_min_list_len` and `find_max_list_len`**: These functions are used to ensure consistency in data preprocessing by identifying the minimum and maximum lengths among samples, facilitating uniform dimensions.  
- **`return_random_audio_sample`**: This function selects a random audio sample index for a specific label, aiding in sampling, debugging, and quick data exploration.  

**Purpose:**
Prepare the CREMA-D dataset for exploratory data analysis and model training. By organizing file paths and labels into the DataFrame `df`, this process simplifies subsequent data processing, feature extraction, and visualization tasks.

## Display Sample Plots

The first 5 audio files from the dataframe `df` correspond to 5 different emotions labeled as **Anger**, **Disgust**, **Fear**, **Happiness**, and **Neutral**, in order.

Using the **librosa** and **librosa.display.waveshow** libraries, we load and display an audio waveform sample for each emotion type. The waveform is a 1D array representing amplitude values, in which: 

- **X-axis**: Represents time.  
- **Y-axis**: Represents amplitude, showing the structure of the audio signal over time.

**Purpose:** Gaining insights into emotional characteristics
- Different emotions often exhibit unique acoustic features such as intensity, pitch variation, and rhythm.  
- Plotting waveforms provides a visual representation of these differences, helping to identify patterns specific to each emotion.

## Preparing Dataset

### Loading Audio into DataFrame

Four empty lists (`X`, `audio_sampling`, `length`, and `audio_arrays`) are initialized to store the results of the audio processing.

The `for` loop iterates over each file path in the `df['path']` column of the dataframe, where each iteration processes one audio file.

The function `librosa.load(i, sr=44100)` loads the audio file at path `i` using a specified sampling rate of 44,100 Hz, which is a standard for high-quality audio.

- `x` is the audio time-series data, represented as a 1D NumPy array of amplitude values. It will be appended to the `X` list, storing the raw audio data for each file.
- `sr` is the sampling rate, which defines the number of samples per second. It will be appended to `audio_sampling`, capturing the rate at which each file was sampled.
- The length of the `x` array, calculated using `len(x)`, will be appended to `length`, recording the duration of each audio file in terms of the number of samples.
- `x` is also appended to `audio_arrays` for further analysis or manipulation.

**Purpose:**
By storing the data in structured lists and arrays, the code prepares the audio data for subsequent steps like audio analysis, feature extraction, and model preprocessing.


### Transforming Labels

- The function `LabelEncoder()` from the `sklearn.preprocessing` library is used to convert categorical labels into numeric form. This transformation is necessary for machine learning models that cannot directly handle non-numeric data.

- After the labels are converted into integers, the function `utils.to_categorical()` from the Keras library is used to transform these integers into a one-hot encoded format. In this format, each label is represented by an array where only one element is 1 (indicating the class label), and all other elements are 0.

**Purpose:** Perform One-Hot Encoding, which is required for training classification models.

- A dictionary, `le_name_mapping`, is created to map each class name (e.g., 'Anger') to its corresponding one-hot encoded vector. The `zip()` function pairs the class names with their one-hot vectors, which are then passed through `dict()` to create the final dictionary.

**Purpose:** Easily reference the one-hot encoded format for each class in the dataset.

#### Results:
```python
{
    'ANG': array([1., 0., 0., 0., 0., 0.]),
    'DIS': array([0., 1., 0., 0., 0., 0.]),
    'FEA': array([0., 0., 1., 0., 0., 0.]),
    'HAP': array([0., 0., 0., 1., 0., 0.]),
    'NEU': array([0., 0., 0., 0., 1., 0.]),
    'SAD': array([0., 0., 0., 0., 0., 1.])
}
```

## Splitting Data into Training, Validation, and Test Sets

The `train_test_split` function from the `sklearn.model_selection` library is used twice to split the dataset into training, validation, and testing sets.

### 1. Splitting the Data into 70% for Training + Validation and 30% for Testing
- `test_size=0.3`: Specifies that 30% of the data should be assigned to the test set, and the remaining 70% will be used for the combined training and validation set.
- `random_state=42`: Ensures reproducibility of the splits.
- `stratify=y`: Ensures that the class distribution in the test set is the same as in the original dataset.

### 2. Splitting the 70% Training + Validation Set into 95% Training and 5% Validation
- `test_size=0.05`: Specifies that 5% of the training+validation set will be used as the validation set, and 95% will be used as the training set.
- `stratify=y[indices_train_val]`: Ensures that the class distribution is maintained for both training and validation sets.

### Results
- **`indices_train_val`**: Indices for the training+validation set (70% of the data). **5209 records**.
- **`indices_test`**: Indices for the test set (30% of the data). **2233 records**.
- **`indices_train`**: Indices for the training set (95% of the training+validation set). **4948 records**.
- **`indices_val`**: Indices for the validation set (5% of the training+validation set). **261 records**.


## Split on Audio Data and Label Data

In this step, we will extract the features (`X`) and labels (`y`) for the training, validation, and testing datasets.

**Purpose:** To ensure that each dataset (training, validation, and testing) contains the appropriate features and labels for its purpose, avoiding overlap and ensuring the model can be trained, validated, and tested on separate data.

### Results
- **`x_train`**: Features for the training set.
- **`x_val`**: Features for the validation set.
- **`x_test`**: Features for the test set.
- **`y_train`**: Labels for the training set.
- **`y_val`**: Labels for the validation set.
- **`y_test`**: Labels for the test set.


## Data Augmentation

Five functions are defined and applied to audio data. Each technique alters the original audio data slightly to create variations, enhancing the diversity of the training dataset.

**1. AWGN (Additive White Gaussian Noise):** This function adds random noise to the audio signal, simulating real-world environments where background noise is always present.

**2. Pitch Adjustment:** This function modifies the pitch of the audio signal without changing its duration, simulating variations in voice tone or pitch that might naturally occur due to different speakers, emotions, or conditions.

**3. Time Stretching:** This function speeds up or slows down the audio signal, changing its duration without altering the pitch, simulating real-world variations where audio may be slightly faster or slower than usual.

**4. Time Shifting:** This function shifts the audio signal in time (circularly), effectively offsetting the waveform, simulating scenarios where the start of the audio signal may not be perfectly aligned.

**5. Data Augmentation:** This function applies multiple data augmentation techniques to a dataset of audio features (`X`) and corresponding labels (`y`). It generates an augmented dataset by retaining the original data and creating new variations using specified augmentation techniques.
  
**Purpose:** This process expands the dataset size, increasing diversity and reducing overfitting.


## Time Domain

### Feature Extraction

This step involves calculating features directly from the raw waveform of audio signals. These features capture the temporal characteristics of the audio.

Key Features:
- **Zero-Crossing Rate (ZCR)**: Measures how often the audio signal's amplitude crosses zero within a frame. ZCR is useful for distinguishing between voiced (low ZCR) and unvoiced (high ZCR) sounds.
- **Root Mean Square Energy (RMSE)**: Measures the energy of the audio signal within a frame. RMSE captures the loudness or intensity of the sound.

The `extract_time_features(X)` function appends the combined feature vector (ZCR + RMSE) for each audio sample to the result list.


### Without Augmentation
- **Process**: The `extract_time_features` function is applied to the original training (`X_train`), validation (`X_val`), and testing (`X_test`) datasets. It generates time-domain feature representations (`X_t_train`, `X_t_val`, `X_t_test`) for each dataset.
- **Purpose**: Extracts temporal features from the raw waveform data to be used in model training, validation, and testing.


### With Augmentation
- **Process**: The `extract_time_features` function is applied to the augmented training dataset (`X_aug_train`), which contains both original and augmented samples.
- **Purpose**: Extracts time-domain features from a more diverse dataset, ensuring the model learns from varied examples.

---

### Padding Feature Space
When working with audio data, padding the feature space in the time domain is crucial to ensure consistent input dimensions. Padding is applied to both the original datasets and the augmented training dataset. It allows the model to preserve the sequential structure of audio signals.

- The `pad_dataset` function is created to pad each audio sample in the dataset X to a specified length m by adding zeros (constant padding) at the beginning and end. The padding is distributed randomly between the start and end of each sample to introduce some variability.

- The training, validation, and test datasets are padded using the above function to ensure all samples have a consistent length.

- The output shows that all samples padded to the length of the longest one (either with or without augmentation), which is 864 data points.

**Purpose:** By standardizing the input length, the padding ensures that the model can process audio signals of varying durations without losing information from shorter sequences.

---
### Mel Spectrograms
- A Mel spectrogram is a visual representation of the frequency content of an audio signal, transformed into a scale that aligns with how humans perceive sound.

- The `mel_spectrogram` function generates Mel spectrograms from audio data by converting time-domain signals into a perceptually meaningful frequency-domain representation. Here, `sr=44100` specifies the default audio sampling rate of 44.1 kHz.

- Creating Mel spectrograms for both original and augmented datasets allows us to assess the model's performance on clean, unmodified data while introducing variability for better generalization.

**Purpose:** Using Mel spectrograms helps the model learn essential patterns from non-augmented data while reducing overfitting to specific characteristics of the original dataset.

---
### Callbacks
- LearningRateScheduler Callback: It is a Keras callback that updates the learning rate for the optimizer at the start of each epoch.

- EarlyStopping Callback: It is used to to stop training early if the model’s performance stops improving on the validation set, in which:
  + `patience=8` means training continues for 8 additional epochs after the last observed improvement. If no further improvement occurs, training stops.
  + `verbose=1`  provides updates about the early stopping process in the console.

**Purpose:** Using a learning rate scheduler together with early stopping provides an effective balance between efficient training and generalization.

---
## Model

- Our model is inspired by the work of Zhao, Jianfeng, Mao, Xia, and Chen, Lijiang, as detailed in their 2019 publication *"Speech Emotion Recognition Using Deep 1D & 2D CNN LSTM Networks" (Biomedical Signal Processing and Control, 47, 312-323)*.

-  The model combines Convolutional Neural Networks (CNNs) and Long Short-Term Memory (LSTM) networks to analyze audio data for emotion recognition.

- 1D CNNs are utilized to extract features from raw time-domain signals, while 2D CNNs focus on spectrogram images, capturing spatial and frequency patterns. These features are processed by LSTM layers to model sequential dependencies, enabling the system to identify complex emotional dynamics effectively. This hybrid approach leverages the strengths of CNNs for feature extraction and LSTMs for sequence analysis, achieving robust performance in speech emotion recognition tasks.

A 1D CNN-LSTM model (`model_1d`) is set up to be trained using the Stochastic Gradient Descent (SGD) optimizer, using the following parameters:
  
  + Learning Rate (`learning_rate=0.0001`): Specifies the step size for updating weights. A small learning rate ensures gradual learning. 
  + Decay (`decay=1e-6`): Gradually decreases the learning rate during training to refine convergence.
  + Momentum (`momentum=0.9`): Helps accelerate SGD in the right direction by smoothing weight updates.
  + Nesterov Momentum (`nesterov=True`): Looks ahead in the gradient direction, improving convergence.

**Purpose:** The use of SGD with Nesterov momentum is advantageous for its ability to escape sharp local minima, improving convergence stability. The decay rate and small learning rate ensure fine-tuning of the model parameters over time.

---

The evaluation of the trained model on the test dataset yields the following results:

**Without Augmentation (With SGD optimizer)**
- Accuracy (0.2984) indicates that the model struggles to generalize on the test data, possibly due to insufficient training.

- Loss (1.7586) suggests that the model's probability distributions for predictions are not strongly aligned with the actual classes, indicating room for improvement in training or model architecture. 

**Without Augmentation (With Adam optimizer)**
- Accuracy (46.02%): The model achieved a higher accuracy (46.02%) compared to the result from the model with the SGD optimizer (29.84%). This shows that the Adam optimizer, with its adaptive learning rate and momentum adjustments, helped the model converge more effectively and improved its ability to generalize to unseen test data.

- Loss (1.3843): The categorical cross-entropy loss on the test set is lower compared to the previous model, indicating that the model's predictions align better with the ground truth.

**With Augementation (with SGD optimizer)**
- Accuracy (18.33%): The model's accuracy is very low (18.33%), which is a significant drop compared to the results with the non-augmented data (46.02% with Adam optimizer). 

- Loss (1.7842): The higher loss value (1.7842) compared to the model with non-augmented data indicates that the model's predictions are further from the true values, suggesting that the augmented data didn't improve the generalization of the model. 

**With Augmentation (with Adam optimizer)**
- Accuracy (48.19%): The accuracy is relatively higher than the previous attempt with SGD but it is still lower than 50%, meaning that it still has room for improvement.

- Loss (1.3373): The loss value is lower than with SGD (1.7842), indicating that the model’s predictions are closer to the true labels. A lower loss signifies that the model is making better predictions overall, even though there is still a large margin of error in its output.

# Frequency Domain


The model in the frequency domain, utilizing a 2D CNN architecture without augmentation, produces the following results:

**Without Augmentation**

- Accuracy (59.99%): It demonstrates that the 2D CNN model effectively extracts features from spectrogram representations of the audio signals.

- Loss (1.0883): A lower loss compared to previous models indicates improved predictions. The categorical cross-entropy loss reflects that the predictions are closer to the true labels.

**With Augmentation**

- Accuracy (62.76%): The model trained with augmented data performs better than the non-augmented version (59.99%). This suggests that data augmentation increases the diversity of training samples, enabling the model to generalize better to unseen data.

- Loss (1.0179): A lower loss reflects that the model's predictions align more closely with the true labels. It indicates fewer significant misclassifications in comparison to the non-augmented model.


**Conclusion:** The frequency-domain models outperformed their time-domain counterparts, with and without augmentation, suggesting that spectrograms provide richer features for emotion recognition. 

---
In this section, we will evaluate the model's predictions against the ground truth across training, validation, and test datasets.

The `compare_performance()` function is used to evaluate performance metrics such as accuracy, precision, recall, F1 score by comparing the true labels (`y_train`, `y_val`, `y_test`) with predicted probabilities or class labels (`y_train_pred_t`, `y_val_pred_t`, `y_test_pred_t`, respectively).

## Time Domain

**Without Augmentation**

- **Training:**

  + Accuracy: The model achieves an overall accuracy of 60% on the training dataset, indicating that the model is moderately effective in classifying speech emotions in the training set. It demonstrates reasonable performance in recognizing some emotions (e.g., Anger and Sadness) but struggles with others like Fear and Disgust.
  + Confusion Matrix Insights: The heatmap shows a strong diagonal trend, which means the model predicts the correct class more often than not for most emotions. It also highlights areas for improvement, especially in reducing misclassifications between similar emotional categories.

- **Validation:**

  + Accuracy score is 42% which is relatively low and reflects the model's moderate ability to classify emotions correctly during validation.
  + Confusion Matrix Insights: The model performs better for emotions like "Anger" and "Sadness" and struggles with emotions such as "Disgust" and "Fear."

- **Test:**

  + Accuracy: The model achieved an accuracy of 45%, highlighting a need for further refinement in model training.
  + Confusion Matrix Insights: The heatmap shows the imbalanced performance across classes, with some performing well and others poorly.

**With Augmentation**

- **Training:**

  + Accuracy score is 53% indicating that that the model correctly classified 53% of the training data. While this is moderate, it shows that the model captures patterns in the augmented training data better than without augmentation.
  + Confusion Matrix Insights: The training performance with augmented data shows moderate improvement compared to non-augmented data. It shows higher values along the diagonal, reflecting correct predictions, particularly for "Anger" (1880 true positives) and "Happiness" (1244 true positives).

- **Validation:**
  + Accuracy score is 45%, meaning that the model correctly predicted 45% of the validation data. This is a moderate improvement compared to the training phase, showing that the model partially generalizes to unseen data.
  + Confusion Matrix Insights: Higher values along the diagonal (e.g., 31 for "Anger" and 30 for "Sadness") indicate correct predictions, but performance varies across emotions.

- **Test:**

  + Accuracy score is 47% indicating that  that the model correctly classified about half of the test dataset. This is a slight improvement over the validation set.
  + Confusion Matrix Insights: Compared to models without augmentation, the recall and precision for classes like "Sadness" have improved, indicating that augmentation helped enhance the model's generalization.

---
  ## Frequency Domain

  **Without Augmentation**

- **Training:**

  + Accuracy score is 53% indicating a strong overall performance during training, showing the model can correctly classify the majority of the training data.
  + Confusion Matrix Insights: The model performs exceptionally well for "Anger" and "Neutral" while "Fear" and "Sadness" show slightly lower performance (around 82-85%) but are still reasonably high. The near-perfect metrics could raise concerns about overfitting, especially without augmentation

- **Validation:**

  + Accuracy score is 59% indicating that the model correctly classifies most validation samples but struggles with some classes. While training accuracy was high (89%), the drop in validation accuracy may indicate slight overfitting.
  + Confusion Matrix Insights: Misclassification occurs between emotions like "Happiness" and "Sadness" or "Disgust" and "Fear", which may share overlapping features in the frequency domain.

- **Test:**

  + Accuracy score is 60% indicating that that the model captures some meaningful patterns in the data but still has room for improvement.
  + Confusion Matrix Insights: While the model performs well on emotions like "Anger," "Neutral," and "Sadness," it struggles with "Disgust" and "Fear." 


**With Augmentation**

- **Training:**

  + Accuracy score is 74% indicating that the model correctly classified nearly three-fourths of the training samples, showing robust performance during training.
  + Confusion Matrix Insights: The augmented frequency-domain model demonstrates strong and balanced performance during training, achieving a robust 74% accuracy with well-distributed precision, recall, and F1-scores. While it excels at recognizing "Anger," "Neutral," and "Happiness," further efforts are needed to address challenges with "Disgust" and "Sadness" for broader generalization.

- **Validation:**

  + Accuracy score is 59%, though not exceptionally high, reflects a balanced performance across most emotion classes but also highlights challenges with certain emotions.
  + Confusion Matrix Insights: The frequency-domain model with augmentation achieves a moderate 59% accuracy on validation data, showing improvements in generalization. While certain emotions like "Anger" and "Sadness" are well-captured, challenges persist in distinguishing subtler emotions like "Disgust" and "Fear."

- **Test:**

  + Accuracy score is 62%, which is higher than the validation accuracy (59%), suggesting effective generalization with augmentation techniques. This improvement demonstrates that augmentation helped the model better understand diverse feature representations in the frequency domain.
  + Confusion Matrix Insights: The frequency-domain model with augmentation achieves a moderate 62% accuracy on test data, reflecting improved generalization. While distinct emotions like "Anger" and "Neutral" are well-captured, subtler emotions like "Disgust" and "Fear" pose challenges.


**Conclusion:**

Although data augmentation enhanced the model's performance in both domains, it still faces somedifficulties in accurately classifying subtle emotions. 