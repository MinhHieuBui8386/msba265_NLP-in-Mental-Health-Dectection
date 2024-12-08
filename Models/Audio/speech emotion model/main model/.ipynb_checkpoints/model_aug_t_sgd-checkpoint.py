import tensorflow as tf
from keras.models import load_model
import librosa
import numpy as np
import random
import math
#model_aug_t_sgd = load_model('C:/Users/Admin/Documents/GitHub/msba265-finalstorage/Models/Audio/speech emotion model/model_aug_t_sgd.keras') 
#model_aug_t_sgd.save('model_aug_t_sgd.h5')

model_aug_t_sgd = tf.keras.models.load_model('C:/Users/Admin/Documents/GitHub/msba265-finalstorage/Models/Audio/speech emotion model/model_aug_t_sgd.keras')
anger_sample,samp = librosa.load('C:/Users/Admin/Documents/GitHub/msba265-finalstorage/data_storage/Audio/CREAMA-D/AudioWAV/1001_DFA_ANG_XX.wav', sr = 44100)
def extract_time_features(X):
    result = []
    for i, sample in enumerate(X):
        zcr = list(librosa.feature.zero_crossing_rate(y=sample.astype(float))[0])
        rmse = list(librosa.feature.rms(y=sample.astype(float))[0])
        zcr += rmse
        result.append(zcr)
    return np.array(result, dtype='object')
def pad_dataset(X, m):
    X_padded = []
    for i, sample in enumerate(X):
        audio_length = len(sample)
        if (audio_length <= m):
            # Length of padding to add at the beginning and end of the signal
            pad_begin_len = random.randint(0, m - audio_length)
            pad_end_len = m - audio_length - pad_begin_len
            X_padded.append(np.pad(sample, (pad_begin_len, pad_end_len), 'constant'))
    
    return np.array(X_padded)

X = []
x, sr = librosa.load('C:/Users/Admin/Documents/GitHub/msba265-finalstorage/data_storage/Audio/CREAMA-D/AudioWAV/1001_DFA_ANG_XX.wav', sr=44100)
X.append(x)
X = np.array(X, dtype='object')
text1 = extract_time_features(X)
X_t_train = pad_dataset(X, 864)
predict = model_aug_t_sgd.predict(X_t_train)
print(predict)