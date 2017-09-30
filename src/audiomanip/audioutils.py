import random
import librosa
import numpy as np
from scipy.stats import mode
from keras import backend as K

import re
import os
import threading

# @Class: AudioUtils
# @Description: Set of methods to handle and transform the data
class AudioUtils(object):
  def __init__(self):
    self.augment_factor = 10

  def random_split(self, x):
    melspec = librosa.feature.melspectrogram(x, n_fft = self.n_fft,
      hop_length = self.hop_length).T

    # Generate a random number
    offset = random.sample(range(0, melspec.shape[0] - 128), 1)[0]
    return melspec[offset:(offset+128)]
  
  def splitsongs_melspect(self, X, y, cnn_type = '1D'):
    temp_X = []
    temp_y = []

    for i, song in enumerate(X):
      song_slipted = np.split(song, self.augment_factor)
      for s in song_slipted:
        temp_X.append(s)
        temp_y.append(y[i])

    temp_X = np.array(temp_X)
    temp_y = np.array(temp_y)

    if not cnn_type == '1D':
      temp_X = temp_X[:, np.newaxis]
    
    return temp_X, temp_y

  def voting(self, y_true, pred):
    if y_true.shape[0] != pred.shape[0]:
      raise ValueError('Both arrays should have the same size!')

    # split the arrays in songs
    arr_size = y_true.shape[0]
    pred = np.split(pred, arr_size/self.augment_factor)
    y_true = np.split(y_true, arr_size/self.augment_factor)

    # Empty answers
    voting_truth = []
    voting_ans = []

    for x,y in zip(y_true, pred):
      voting_truth.append(mode(x)[0][0])
      voting_ans.append(mode(y)[0][0])
    
    return np.array(voting_truth), np.array(voting_ans)

# @Class: MusicDataGenerator
# @Description:
#   featurewise_center: set input mean to 0 over the dataset.
#   samplewise_center: set each sample mean to 0.
#   featurewise_std_normalization: divide inputs by std of the dataset.
#   samplewise_std_normalization: divide each input by its std.
#   zca_whitening: apply ZCA whitening.      
class MusicDataGenerator(object):
  def __init__(self,
    time_stretching=False,
    pitch_shifting=False,
    dynamic_range_compression=False,
    background_noise=False):

    self.time_stretching = time_stretching
    self.pitch_shifting = pitch_shifting
    self.background_noise = background_noise
    self.dynamic_range_compression = dynamic_range_compression
    self.audioutils = AudioUtils()

  def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None):
    return NumpyArrayIterator(
      x, y, self,
      batch_size=batch_size,
      shuffle=shuffle,
      seed=seed)

  def random_transform(self, x, seed=None):
    if seed is not None:
      np.random.seed(seed)

    # Random transformations on the audio file
    # Timestretching
    if self.time_stretching:
      factor = np.random.uniform(0.8,1.2)
      x = librosa.effects.time_stretch(x, factor)
    
    # Pitch shifting
    if self.pitch_shifting:
      passif
    
    # Add Background Noise
    if self.background_noise:
      passif

    # Execute dynamic range compression
    if self.dynamic_range_compression:
      passif

    # Get a random sample of the melspectrogram
    x = self.audioutils.random_split(x)
    return x

  def fit(self, x, augment=False, rounds=1, seed=None):
    x = np.asarray(x, dtype=K.floatx())
    
    if x.ndim != 2:
      raise ValueError('Input to `.fit()` should have rank 2. '
        'Got array with shape: ' + str(x.shape))
    
    if seed is not None:
      np.random.seed(seed)

    x = np.copy(x)
    if augment:
      ax = np.zeros(tuple([rounds * x.shape[0]] + list(x.shape)[1:]), dtype=K.floatx())
      for r in range(rounds):
        for i in range(x.shape[0]):
          ax[i + r * x.shape[0]] = self.random_transform(x[i])
      x = ax

# @Class: Iterator
# @Description:
#   Abstract base class for Music data iterators.
#   n: Integer, total number of samples in the dataset to loop over.
#   batch_size: Integer, size of a batch.
#   shuffle: Boolean, whether to shuffle the data between epochs.
#   seed: Random seeding for data shuffling.
class Iterator(object):
  def __init__(self, n, batch_size, shuffle, seed):
    self.n = n
    self.batch_size = batch_size
    self.shuffle = shuffle
    self.batch_index = 0
    self.total_batches_seen = 0
    self.lock = threading.Lock()
    self.index_generator = self._flow_index(n, batch_size, shuffle, seed)

  def reset(self):
    self.batch_index = 0

  def _flow_index(self, n, batch_size=32, shuffle=False, seed=None):
    # Ensure self.batch_index is 0.
    self.reset()
    while 1:
      if seed is not None:
        np.random.seed(seed + self.total_batches_seen)
      if self.batch_index == 0:
        index_array = np.arange(n)
        if shuffle:
          index_array = np.random.permutation(n)

      current_index = (self.batch_index * batch_size) % n
      if n > current_index + batch_size:
        current_batch_size = batch_size
        self.batch_index += 1
      else:
        current_batch_size = n - current_index
        self.batch_index = 0
      self.total_batches_seen += 1
      yield (index_array[current_index: current_index + current_batch_size],
        current_index, current_batch_size)

  def __iter__(self):
    # Needed if we want to do something like:
    # for x, y in data_gen.flow(...):
    return self

  def __next__(self, *args, **kwargs):
    return self.next(*args, **kwargs)

# @Class: NumpyArrayIterator
# @Description:
#   Iterator yielding data from a Numpy array.
#   x: Numpy array of input data.
#   y: Numpy array of targets data.
#   music_data_generator: Instance of `MusicDataGenerator`
#     to use for random transformations.
#   batch_size: Integer, size of a batch.
#   shuffle: Boolean, whether to shuffle the data between epochs.
#   seed: Random seed for data shuffling.
class NumpyArrayIterator(Iterator):
  def __init__(self, x, y, music_data_generator,
    batch_size=32, shuffle=False, seed=None):
    if y is not None and len(x) != len(y):
      raise ValueError('X (music tensor) and y (labels) '
        'should have the same length. '
        'Found: X.shape = %s, y.shape = %s' %
        (np.asarray(x).shape, np.asarray(y).shape))

    self.x = np.asarray(x, dtype=K.floatx())
    if self.x.ndim != 2:
      raise ValueError('Input data in `NumpyArrayIterator` '
        'should have rank 2. You passed an array '
        'with shape', self.x.shape)
    
    if y is not None:
      self.y = np.asarray(y)
    else:
        self.y = None
    self.music_data_generator = music_data_generator
    super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

  def next(self):
    # Keeps under lock only the mechanism which advances
    # the indexing of each batch.
    with self.lock:
      index_array, current_index, current_batch_size = next(self.index_generator)
    
    # The transformation of images is not under thread lock
    # so it can be done in parallel
    melspec_size = [128,128]
    batch_x = np.zeros(tuple([current_batch_size] + melspec_size), dtype=K.floatx())
    for i, j in enumerate(index_array):
      x = self.x[j]
      x = self.music_data_generator.random_transform(x.astype(K.floatx()))
      batch_x[i] = x
    
    if self.y is None:
      return batch_x
    batch_y = self.y[index_array]
    return batch_x, batch_y
