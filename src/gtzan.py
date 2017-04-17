import sys
import argparse
import numpy as np

import keras
from keras.models import Sequential
from keras.layers import Dense, Dropout, Flatten
from keras import backend as K

from mfcc import MFCC
from spectrogram import MelSpectrogram

# Constants
GTZAN_FOLDER = '../dataset/GTZAN/'

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
  elif args.rep == 'SPECT':
    # Create a MFCC representation from the GTZAN Dataset
    song_rep = MelSpectrogram(GTZAN_FOLDER)
  else:
    # Invalid option selected
    raise ValueError('Argument Invalid: The options are MFCC or SPECT')

  songs, genres = song_rep.getdata()
  print(songs)

if __name__ == "__main__":
  main(sys.argv)