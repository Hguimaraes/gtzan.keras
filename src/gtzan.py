import os
import gc
import logging
import argparse

# Disable TF warnings about speed up and future warnings
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

# Disable warnings from h5py
import warnings
with warnings.catch_warnings():
    warnings.filterwarnings("ignore", category = FutureWarning)
    import h5py

# Audio processing and DL frameworks 
import librosa
import numpy as np
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score

import keras
from keras import backend as K

from gtzan import *

# Constants


def main(args):
    # Validate arguments
    exec_mode = ['train', 'test']
    if args.type not in exec_mode:
        raise ValueError("Invalid type parameter. Should be 'train' or 'test'.")

    # Start
    if args.type == 'train':
        print("train")
    else:
        print("test")

if __name__ == '__main__':
    # Parse command line arguments
    parser = argparse.ArgumentParser(description='Music Genre Recognition on GTZAN')

    # Required arguments
    parser.add_argument('-t', '--type', help='train or test mode to execute', type=str, required=True)

    # Nearly optional arguments. Should be filled according to the option of the requireds
    parser.add_argument('-d', '--directory', help='Path to the root directory with GTZAN files', type=str)
    parser.add_argument('--model', help='If choosed test, path to trained model', type=str)
    args = parser.parse_args()

    # Call the main function
    main(args)