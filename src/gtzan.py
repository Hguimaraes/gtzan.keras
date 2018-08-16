import os
import gc
import logging
import argparse
from datetime import datetime

# Disable TF warnings about speed up and future warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

# Disable warnings from h5py
import warnings
warnings.filterwarnings("ignore", category = FutureWarning)

# Audio processing and DL frameworks 
import h5py
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.metrics import accuracy_score

import keras
from keras import backend as K

from gtzan import *

# Constants
song_samples = 660000
genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4, 
          'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
num_genres = len(genres)

def main(args):
    exec_mode = ['train', 'test']
    exec_time = datetime.now().strftime('%Y%m%d%H%M%S')

    # Validate arguments
    if args.type not in exec_mode:
        raise ValueError("Invalid type parameter. Should be 'train' or 'test'.")

    # Start
    if args.type == 'train':
        # Check if the directory path to GTZAN files was inputed
        if not args.directory:
            raise ValueError("File path to model should be passed in test mode.")

        # Read the files to memory and split into train test
        X, y = read_data(args.directory, genres, song_samples)
        X_train, X_test, y_train, y_test = ttsplit(X, y, test_size=0.3)

        # Histogram for train and test 
        values, count = np.unique(np.argmax(y_train, axis=1), return_counts=True)
        plt.bar(values, count)

        values, count = np.unique(np.argmax(y_test, axis=1), return_counts=True)
        plt.bar(values, count)
        plt.savefig('../logs/{}/histogram.png'.format(exec_time),
            format='png', bbox_inches='tight')

        # Training step
        #input_shape = X_train[0].shape
        #cnn = build_model(input_shape, num_genres)
        #cnn.compile(loss=keras.losses.categorical_crossentropy,
        #      optimizer=keras.optimizers.Adam(),
        #      metrics=['accuracy'])

        #hist = cnn.fit(X_train, y_train,
        #          batch_size=32,
        #          epochs=150,
        #          verbose=1,
        #          validation_data=(X_test, y_test))

        # Evaluate
        #score = cnn.evaluate(X_test, y_test, verbose=0)
        #print("val_loss = {:.3f} and val_acc = {:.3f}".format(score[0], score[1]))

        # Plot graphs
        #save_history(hist, '../logs/evaluate.png')

        # Save the confusion Matrix
        #preds = np.argmax(cnn.predict(X_test), axis = 1)
        #y_orig = np.argmax(y_test, axis = 1)
        #cm = confusion_matrix(preds, y_orig)

        #keys = OrderedDict(sorted(genres.items(), key=lambda t: t[1])).keys()

        #plt.figure(figsize=(8,8))
        #plot_confusion_matrix(cm, keys, normalize=True)

        # Save the model

    else:
        # Check if the file path to the model was passed
        if not args.model:
            raise ValueError("File path to model should be passed in test mode.")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Music Genre Recognition on GTZAN')

    # Required arguments
    parser.add_argument('-t', '--type', help='train or test mode to execute', type=str, required=True)

    # Nearly optional arguments. Should be filled according to the option of the requireds
    parser.add_argument('-d', '--directory', help='Path to the root directory with GTZAN files', type=str)
    parser.add_argument('-m', '--model', help='If choosed test, path to trained model', type=str)
    args = parser.parse_args()

    # Call the main function
    main(args)