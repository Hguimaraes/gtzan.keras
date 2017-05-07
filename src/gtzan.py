import os
import sys
import argparse
import numpy as np
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit

import keras
from keras import backend as K

from mfcc import MFCC
from spectrogram import MelSpectrogram
from models import cnn_gtzan_model

# Disable TF warnings about speed up
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Constants
EXEC_TIMES = 15
GTZAN_FOLDER = '../dataset/GTZAN/'
batch_size = 64
epochs = 30

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
  songs = song_rep.normalize(songs)
  print(songs.shape)
  print(genres.shape)

  # Free memory
  del song_rep
  
  # Train multiple times and get mean score
  test_loss = []
  test_acc = []

  for x in range(EXEC_TIMES):
    # Split the dataset into training and test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
    for train_index, test_index in sss.split(songs, genres):
      x_train, x_test = songs[train_index], songs[test_index]
      y_train, y_test = genres[train_index], genres[test_index]

    # Construct the model
    cnn = cnn_gtzan_model(input_shape)
    print("Size of the CNN: %s" % cnn.count_params())

    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True)
    cnn.compile(loss=keras.losses.categorical_crossentropy,
      optimizer=sgd,
      metrics=['accuracy'])

    cnn.fit(x_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      validation_data=(x_test, y_test))

    score = cnn.evaluate(x_test, y_test, verbose=0)
    
    # Save metrics to calculate the mean
    test_loss.append(score[0])
    test_acc.append(score[1])

    # Print the confusion matrix of the model
    pred_values = np.argmax(cnn.predict(x_test), axis = 1)
    cm = confusion_matrix(np.argmax(y_test, axis = 1), pred_values)
    print(cm)

    # Print metrics
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

  # Print the statistics
  print("Test accuracy - mean: %s, std: %s" % (np.mean(test_acc), np.std(test_acc)))

if __name__ == "__main__":
  main(sys.argv)