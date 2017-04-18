import os
import sys
import argparse
import numpy as np
from sklearn.model_selection import StratifiedShuffleSplit
from sklearn.preprocessing import OneHotEncoder

import keras
from keras.models import Sequential
from keras.utils import plot_model
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras import backend as K

from mfcc import MFCC
from spectrogram import MelSpectrogram

# Disable TF warnings about speed up
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Constants
GTZAN_FOLDER = '../dataset/GTZAN/'
batch_size = 32 
epochs = 100

"""
"""
def cnn_gtzan_model(input_shape):
  model = Sequential()
  
  # First Conv Layer
  model.add(Conv1D(256,
    kernel_size = 8,
    activation='relu',
    input_shape = input_shape))
  model.add(MaxPooling1D(pool_size=8, strides=8))
  
  # Second Conv Layer
  model.add(Conv1D(512, 8, activation='relu'))
  model.add(GlobalMaxPooling1D())

  # Regular MLP
  model.add(Dense(1024, activation='relu'))
  model.add(Dropout(0.25))
  model.add(Dense(10, activation='softmax'))

  return model

"""
"""
def main(argv):
  # Pass argument
  parser = argparse.ArgumentParser()
  parser.add_argument("rep", help="MFCC || SPECT: Choose to use MFCC or Spectrogram")
  args = parser.parse_args()

  if args.rep == 'MFCC':
    # Create a melspectrogram from the GTZAN Dataset
    song_rep = MFCC(GTZAN_FOLDER)
    input_shape = (1290, 20)
  elif args.rep == 'SPECT':
    # Create a MFCC representation from the GTZAN Dataset
    song_rep = MelSpectrogram(GTZAN_FOLDER)
    input_shape = (1290, 128)
  else:
    # Invalid option selected
    raise ValueError('Argument Invalid: The options are MFCC or SPECT')

  songs, genres = song_rep.getdata()
  print(songs.shape)
  print(genres.shape)
  
  # Split the dataset into training and test
  sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3, random_state=0)
  for train_index, test_index in sss.split(songs, genres):
    x_train, x_test = songs[train_index], songs[test_index]
    y_train, y_test = genres[train_index], genres[test_index]

  # Construct the model
  cnn = cnn_gtzan_model(input_shape)
  print("Size of the CNN: %s" % cnn.count_params())
  print("Network summary:\n %s" % cnn.summary())

  cnn.compile(loss=keras.losses.categorical_crossentropy,
    optimizer=keras.optimizers.Adadelta(),
    metrics=['accuracy'])

  cnn.fit(x_train, y_train,
    batch_size=batch_size,
    epochs=epochs,
    verbose=1,
    validation_data=(x_test, y_test))

  score = cnn.evaluate(x_test, y_test, verbose=0)
  print('Test loss:', score[0])
  print('Test accuracy:', score[1])

if __name__ == "__main__":
  main(sys.argv)