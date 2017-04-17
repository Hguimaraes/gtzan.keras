import os
import librosa
import numpy as np

"""
"""
class MelSpectrogram(object):
  def __init__(self, file_path):
    # Constants
    self.song_samples = 660000
    self.n_fft = 2048
    self.hop_length = 512
    self.file_path = file_path
    self.genres = ['metal', 'disco', 'classical', 'hip-hop', 'jazz', 'country',
      'pop', 'blues', 'reggae', 'rock']
  
  def getdata(self, frac = 0.75):
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
          melspec = librosa.feature.melspectrogram(signal[:self.song_samples],
            sr = sr, n_fft = self.n_fft, hop_length = self.hop_length)

          # Append the result to the data structure
          song_data.append(melspec)
          genre_data.append(x)
    return song_data, genre_data