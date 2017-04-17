import os
import librosa
import numpy as np

"""
"""
class MFCC(object):
  def __init__(self, file_path):
    # Constants
    self.song_samples = 660000
    self.n_mfcc = 20
    self.file_path = file_path
    self.genres = ['metal', 'disco', 'classical', 'hiphop', 'jazz', 'country',
      'pop', 'blues', 'reggae', 'rock']
  
  def getdata(self):
    # Allocate memory for the array of songs
    song_data = []
    genre_data = []

    # Read files from the folders
    for x in self.genres:
      for root, subdirs, files in os.walk(self.file_path + x):
        for file in files:
          # Read the audio file
          file_name = self.file_path + x + "/" + file
          print('READING: %s' % file_name)
          signal, sr = librosa.load(file_name)
          
          # Calculate the melspectrogram of it
          mfcc = librosa.feature.mfcc(signal[:self.song_samples],
            sr = sr, n_mfcc = self.n_mfcc)

          # Append the result to the data structure
          song_data.append(mfcc)
          genre_data.append(x)
    return np.array(song_data), np.array(genre_data)