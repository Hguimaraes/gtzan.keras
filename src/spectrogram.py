import os
import keras
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
    self.genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
     'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
  
  def getdata(self):
    # Allocate memory for the array of songs
    song_data = []
    genre_data = []

    # Read files from the folders
    for x,_ in self.genres.items():
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
          song_data.append(np.transpose(melspec))
          genre_data.append(self.genres[x])
    return np.array(song_data), keras.utils.to_categorical(genre_data, len(self.genres))