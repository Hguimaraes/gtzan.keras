import os
import keras
import librosa
import numpy as np

# @Class: MFCC
# @Description: 
#  Class to read .au files and export the songs as MFCCs
class MFCC(object):
  def __init__(self, file_path):
    # Constants
    self.song_samples = 660000
    self.n_mfcc = 20
    self.tol = 1e-3
    self.file_path = file_path
    self.genres = {'metal': 0, 'disco': 1, 'classical': 2, 'hiphop': 3, 'jazz': 4,
     'country': 5, 'pop': 6, 'blues': 7, 'reggae': 8, 'rock': 9}
  
  # @Method: getdata
  # @Description:
  #  Retrieve data from .au files and return then as numpy arrays
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

# @Class: MelSpectrogram
# @Description: 
#  Class to read .au files and export the songs as MelSpectrograms
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
   
  # @Method: getdata
  # @Description:
  #  Retrieve data from .au files and return then as numpy arrays
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
            
            # Split in 15 pieces
            melspec = np.split(melspec.T[:1280,:], 10)

            # Append the result to the data structure
            for m in melspec:
              song_data.append(m)
              genre_data.append(self.genres[x])
    return np.array(song_data), keras.utils.to_categorical(genre_data, len(self.genres))