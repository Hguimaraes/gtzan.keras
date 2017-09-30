import gc
import os
import ast
import sys
import configparser

import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
import keras
from keras import backend as K

from audiomanip.audiostruct import AudioStruct
from audiomanip.audiomodels import ModelZoo
from audiomanip.audioutils import AudioUtils
from audiomanip.audioutils import MusicDataGenerator

# Disable TF warnings about speed up
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

def main():
  # Parse config file
  config = configparser.ConfigParser()
  config.read('params.ini')

  # Constants
  ## Configuration
  GTZAN_FOLDER = config['FILE_READ']['GTZAN_FOLDER']
  MODEL_PATH = config['FILE_READ']['SAVE_MODEL']
  SAVE_NPY = ast.literal_eval(config['FILE_READ']['SAVE_NPY'])
  EXEC_TIMES = int(config['PARAMETERS_MODEL']['EXEC_TIMES'])
  CNN_TYPE = config['PARAMETERS_MODEL']['CNN_TYPE']

  ## CNN hyperparameters
  batch_size = int(config['PARAMETERS_MODEL']['BATCH_SIZE'])
  epochs = int(config['PARAMETERS_MODEL']['EPOCHS'])

  if not ((CNN_TYPE == '1D') or (CNN_TYPE == '2D')):
    raise ValueError('Argument Invalid: The options are 1D or 2D for CNN_TYPE')

  # Read data
  data_type = config['FILE_READ']['TYPE']
  input_shape = (128, 128)
  print("data_type: %s" % data_type)

  ## Read the .au files
  if data_type == 'AUDIO_FILES':
    song_rep = AudioStruct(GTZAN_FOLDER)
    songs, genres = song_rep.getdata()

    # Save the audio files as npy files to read faster next time
    if SAVE_NPY:
      np.save(GTZAN_FOLDER + 'songs.npy', songs)
      np.save(GTZAN_FOLDER + 'genres.npy', genres)

  ## Read from npy file
  elif data_type == 'NPY':
    songs = np.load(GTZAN_FOLDER + 'songs.npy')
    genres = np.load(GTZAN_FOLDER + 'genres.npy')
  
  ## Not valid datatype
  else:
    raise ValueError('Argument Invalid: The options are AUDIO_FILES or NPY for data_type')

  print("Original songs array shape: {0}".format(songs.shape))
  print("Original genre array shape: {0}".format(genres.shape))

  # Train multiple times and get mean score
  val_acc = []
  test_history = []
  test_acc = []
  test_acc_mvs = []

  for x in range(EXEC_TIMES):
    # Split the dataset into training and test
    X_train, X_test, y_train, y_test = train_test_split(
      songs, genres, test_size=0.1, stratify=genres)
    
    # Split training set into training and validation
    X_train, X_Val, y_train, y_val = train_test_split(
      X_train, y_train, test_size=1/6, stratify=y_train)
        
    # split the train, test and validation data in size 128x128
    X_Val, y_val = AudioUtils().splitsongs_melspect(X_Val, y_val, CNN_TYPE)
    X_test, y_test = AudioUtils().splitsongs_melspect(X_test, y_test, CNN_TYPE)
    X_train, y_train = AudioUtils().splitsongs_melspect(X_train, y_train, CNN_TYPE)
    
    # Construct the model
    if CNN_TYPE == '1D':
      cnn = ModelZoo.cnn_melspect_1D(input_shape)
    elif CNN_TYPE == '2D':
      cnn = ModelZoo.cnn_melspect_2D(input_shape)

    print("\nTrain shape: {0}".format(X_train.shape))
    print("Validation shape: {0}".format(X_Val.shape))
    print("Test shape: {0}\n".format(X_test.shape))
    print("Size of the CNN: %s\n" % cnn.count_params())

    # Optimizers
    sgd = keras.optimizers.SGD(lr=0.001, momentum=0.9, decay=1e-5, nesterov=True)
    adam = keras.optimizers.Adam(lr=0.001, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=1e-5)
    
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
    history = cnn.fit(X_train, y_train,
      batch_size=batch_size,
      epochs=epochs,
      verbose=1,
      validation_data=(X_Val, y_val),
      callbacks = [earlystop])

    score = cnn.evaluate(X_test, y_test, verbose=0)
    score_val = cnn.evaluate(X_Val, y_val, verbose=0)
        
    # Majority Voting System
    pred_values = np.argmax(cnn.predict(X_test), axis = 1)
    mvs_truth, mvs_res = AudioUtils().voting(np.argmax(y_test, axis = 1), pred_values)
    acc_mvs = accuracy_score(mvs_truth, mvs_res)

    # Save metrics
    val_acc.append(score_val[1])
    test_acc.append(score[1])
    test_history.append(history)
    test_acc_mvs.append(acc_mvs)

    # Print metrics
    print('Test accuracy:', score[1])
    print('Test accuracy for Majority Voting System:', acc_mvs)

    # Print the confusion matrix for Voting System
    cm = confusion_matrix(mvs_truth, mvs_res)
    print(cm)

  # Print the statistics
  print("Validation accuracy - mean: %s, std: %s" % (np.mean(val_acc), np.std(val_acc)))
  print("Test accuracy - mean: %s, std: %s" % (np.mean(test_acc), np.std(test_acc)))
  print("Test accuracy MVS - mean: %s, std: %s" % (np.mean(test_acc_mvs), np.std(test_acc_mvs)))
  
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

  # Save the model
  cnn.save(MODEL_PATH)

  # Free memory
  del songs
  del genres
  gc.collect()

if __name__ == '__main__':
  main()