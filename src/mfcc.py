import os
import keras
import librosa
import numpy as np

"""
"""
class MFCC(object):
  def __init__(self, file_path):
    # Constants
    self.song_samples = 660000
    self.n_mfcc = 20
    self.tol = 1e-3
    self.file_path = file_path
    self.genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
     'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
  
  """
  """
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
          mfcc = librosa.feature.mfcc(signal[:self.song_samples],
            sr = sr, n_mfcc = self.n_mfcc)

          # Append the result to the data structure
          song_data.append(np.transpose(mfcc))
          genre_data.append(self.genres[x])
    return np.array(song_data), keras.utils.to_categorical(genre_data, len(self.genres))

  """
  """
  def normalize(self, songs):
    # Allocate memory
    norm_songs = np.zeros(songs.shape)
    for i in range(songs.shape[0]):
      # Subtrac the mean
      song_mean_channel = list(map(lambda x, y: x - y, songs[i], np.mean(songs[i], axis=1)))
      song_mean_channel = np.array(song_mean_channel)
        
      # Get the std of each channel
      song_std = np.std(songs[i], axis=1)
      song_std[song_std <= self.tol] = 1
        
      # Division by the std
      song_norm_channel = list(map(lambda x, y: x/y, song_mean_channel, song_std))
      song_norm_channel = np.array(song_norm_channel)
        
      # Save normalized spectrograms
      norm_songs[i] = song_norm_channel
    return norm_songs