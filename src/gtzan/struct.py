import os
import librosa
import numpy as np
from sklearn.model_selection import train_test_split
from keras.utils import to_categorical

"""
@description: Method to split a song into multiple songs using overlapping windows
"""
def splitsongs(X, y, window = 0.1, overlap = 0.5):
    # Empty lists to hold our results
    temp_X = []
    temp_y = []

    # Get the input song array size
    xshape = X.shape[0]
    chunk = int(xshape*window)
    offset = int(chunk*(1.-overlap))
    
    # Split the song and create new ones on windows
    spsong = [X[i:i+chunk] for i in range(0, xshape - chunk + offset, offset)]
    for s in spsong:
        temp_X.append(s)
        temp_y.append(y)

    return np.array(temp_X), np.array(temp_y)

"""
@description: Method to convert a list of songs to a np array of melspectrograms
"""
def to_melspectrogram(songs, n_fft = 1024, hop_length = 512):
    # Transformation function
    melspec = lambda x: librosa.feature.melspectrogram(x, n_fft = n_fft,
        hop_length = hop_length)[:,:,np.newaxis]

    # map transformation of input songs to melspectrogram using log-scale
    tsongs = map(melspec, songs)
    return np.array(list(tsongs))

"""
@description: Read audio files from folder
"""
def read_data(src_dir, genres, song_samples,  
    n_fft = 1024, hop_length = 512, debug = True):
    # Empty array of dicts with the processed features from all files
    arr_specs = []
    arr_genres = []

    # Read files from the folders
    for x, _ in genres.items():
        folder = src_dir + x
        
        for root, subdirs, files in os.walk(folder):
            for file in files:
                # Read the audio file
                file_name = folder + "/" + file
                signal, sr = librosa.load(file_name)
                signal = signal[:song_samples]
                
                # Debug process
                if debug:
                    print("Reading file: {}".format(file_name))
                
                # Convert to dataset of spectograms/melspectograms
                signals, y = splitsongs(signal, genres[x])
                
                # Convert to "spec" representation
                specs = to_melspectrogram(signals, n_fft, hop_length)
                
                # Save files
                arr_genres.append(y)
                arr_specs.append(specs)
                
    return np.array(arr_specs), np.array(arr_genres)

"""
@description: Split train and test in chunks
"""
def ttsplit(X, y, test_size):
    # Stratify array
    strat_y = np.max(y, axis = 1)
    s = np.arange(y.shape[0])
    
    # Shuffle the arrays
    np.random.shuffle(s)
    X, y = X[s], y[s]

    # Split the arrays
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = test_size, stratify = strat_y)

    # Flatenizer the arrays
    X_train = X_train.reshape(-1, *X_train.shape[-3:])
    X_test = X_test.reshape(-1, *X_test.shape[-3:])
    y_train = to_categorical(y_train.reshape(-1))
    y_test = to_categorical(y_test.reshape(-1))

    return X_train, X_test, y_train, y_test