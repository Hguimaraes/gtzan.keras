import gc
import logging
import argparse
from datetime import datetime
from collections import OrderedDict
# Disable TF warnings about speed up and future warnings
#os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable warnings from h5py
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)
import math
# Audio processing and DL frameworks 
import h5py
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score
from sklearn.model_selection import train_test_split

import keras
from keras import backend as K
from keras.models import load_model

genres = {0: 'metal', 1 : 'disco', 2: 'classical', 3 :'hiphop', 4:'jazz', 
          5 :'country', 6:'pop', 7:'blues', 8:'reggae', 9:'rock'}

song_samples = 660000

def load_song(filepath):
    y, sr = librosa.load(filepath)
    y = y[:song_samples]
    return y, sr

def splitsongs(X, window = 0.1, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []
    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)

    return np.array(temp_X)
#print(splitsongs(y))

def to_melspec(signals):
    #mel_spec = [librosa.feature.melspectrogram(i) for i in signals]
    melspec = lambda x : librosa.feature.melspectrogram(x, sr=22050, n_fft=1024, hop_length=512)[:, :, np.newaxis]
    spec_array = map(melspec, signals)
    return np.array(list(spec_array))


def evaluate(filepath):
    y = load_song(filepath)[0]
    predictions = []
    spectro = []
    signals = splitsongs(y)
    spec_array = to_melspec(signals)
    spectro.extend(spec_array)
    spectro = np.array(spectro)
    spectro = np.squeeze(np.stack((spectro,)*3,-1)) #In case you're using VGG16
    print(spectro.shape)
    model = keras.models.load_model("Enter Model Path Here")
    #print(model.summary())
    #for i in range(len(spec_array)):
    predictions = np.array(model.predict(spectro))
    preds = np.argmax(predictions, axis=1)
    print(genres[np.bincount(preds).argmax()])


print(evaluate(input("Enter File Path :")))        


