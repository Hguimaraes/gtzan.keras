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
    self.tol = 10e-3
    self.file_path = file_path
    self.genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
      'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
  
  def getdata(self):
    # Structure for the array of songs
    song_data = []
    genre_data = []
        
    # Read files from the folders
    for x,_ in self.genres.items():
      for root, subdirs, files in os.walk(self.file_path + x):
        for file in files:
          # Read the audio file
            file_name = self.file_path + x + "/" + file
            print(file_name)
            signal, sr = librosa.load(file_name)
          
            # Calculate the melspectrogram of the audio and use log scale
            melspec = librosa.feature.melspectrogram(signal[:self.song_samples],
              sr = sr, n_fft = self.n_fft, hop_length = self.hop_length)
          
            # Append the result to the data structure
            song_data.append(melspec.T)
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
  
  """
  """
  def logscale(self, songs):
    # Allocate memory
    logscale_songs = np.zeros(songs.shape)
    for i in range(songs.shape[0]):
      logscale_songs[i] = librosa.logamplitude(songs[i])
    return logscale_songs