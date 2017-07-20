import os
import sys
import configparser

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import StratifiedShuffleSplit
import keras
from keras import backend as K

from audiostruct import MFCC, MelSpectrogram
from audiomodels import ModelZoo
#from audioutils import MusicDataGenerator

# Disable TF warnings about speed up
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
  # Parse config file
  config = configparser.ConfigParser()
  config.read('params.ini')

  # Constants
  folder = config['FILE_READ']['GTZAN_FOLDER']
  EXEC_TIMES = int(config['PARAMETERS_MODEL']['EXEC_TIMES'])
  batch_size = int(config['PARAMETERS_MODEL']['BATCH_SIZE'])
  epochs = int(config['PARAMETERS_MODEL']['EPOCHS'])

  # Read data
  data_type = config['FILE_READ']['TYPE']
  print("data_type: %s" % data_type)

  ## Read as MFCC
  if data_type == 'MFCC':
    input_shape = (1290, 128)
    song_rep = MFCC(GTZAN_FOLDER)
    songs, genres = song_rep.getdata()

  ## Read as MelSpectrogram
  elif data_type == 'SPECT':
    input_shape = (1290, 20)
    song_rep = MelSpectrogram(GTZAN_FOLDER)
    songs, genres = song_rep.getdata()

  ## Read from npy file
  elif data_type == 'NPY':
    input_shape = (1290, 128)
    songs = np.load(folder + 'songs.npy')
    genres = np.load(folder + 'genres.npy')
  
  ## Not valid datatype
  else:
    raise ValueError('Argument Invalid: The options are MFCC, SPECT or NPY')

  print(songs.shape)
  print(genres.shape)
  
  # Train multiple times and get mean score
  test_history = []
  test_loss = []
  test_acc = []

  for x in range(EXEC_TIMES):
    # Split the dataset into training and test
    sss = StratifiedShuffleSplit(n_splits=1, test_size=0.3)
    for train_index, test_index in sss.split(songs, genres):
      x_train, x_test = songs[train_index], songs[test_index]
      y_train, y_test = genres[train_index], genres[test_index]
    
    # Construct the model
    if data_type == 'MFCC':
      cnn = ModelZoo.cnn_mfcc(input_shape)
    else:
      cnn = ModelZoo.cnn_melspect(input_shape)

    print("Size of the CNN: %s" % cnn.count_params())

    # Optimizers
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-6, nesterov=True)
    # Compiler for the model
    cnn.compile(loss=keras.losses.categorical_crossentropy,
      optimizer=sgd,
      metrics=['accuracy'])

    # Early stop
    earlystop = keras.callbacks.EarlyStopping(monitor='val_loss',
      min_delta=0,
      patience=2,
      verbose=0,
      mode='auto')

    # Fit the model
    history = cnn.fit(x_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      validation_data=(x_test, y_test),
      callbacks = [earlystop])

    score = cnn.evaluate(x_test, y_test, verbose=0)
    
    # Save metrics to calculate the mean
    test_loss.append(score[0])
    test_acc.append(score[1])
    test_history.append(history)

    # Print the confusion matrix of the model
    pred_values = np.argmax(cnn.predict(x_test), axis = 1)
    cm = confusion_matrix(np.argmax(y_test, axis = 1), pred_values)
    print(cm)

    # Print metrics
    print('Test loss:', score[0])
    print('Test accuracy:', score[1])

  # Print the statistics
  print(list(test_acc))
  print("Test accuracy - mean: %s, std: %s" % (np.mean(test_acc), np.std(test_acc)))
  
  # summarize history for accuracy
  plt.plot(history.history['acc'])
  plt.plot(history.history['val_acc'])
  plt.title('model accuracy')
  plt.ylabel('accuracy')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()
  
  # summarize history for loss
  plt.plot(history.history['loss'])
  plt.plot(history.history['val_loss'])
  plt.title('model loss')
  plt.ylabel('loss')
  plt.xlabel('epoch')
  plt.legend(['train', 'test'], loc='upper left')
  plt.show()

if __name__ == '__main__':
  main()