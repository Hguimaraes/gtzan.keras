import os
import keras
import librosa
import numpy as np

class MusicDataGenerator(object):
  def __init__(self,
               time_stretching=False,
               pitch_shifting=False,
               dynamic_range_compression=False,
               background_noise=False):
      
    self.time_stretching = time_stretching
    self.pitch_shifting = pitch_shifting
    self.dynamic_range_compression = dynamic_range_compression
    self.background_noise = background_noise
    
    self.mean = None
    self.std = None
      
  def flow(self, x, y=None, batch_size=32, shuffle=True, seed=None):
    return NumpyArrayIterator(
          x, y, self,
          batch_size=batch_size,
          shuffle=shuffle,
          seed=seed)

  def standardize(self, x):
    return x

  def random_transform(self, x, seed=None):
    return x

  def fit(self, x, augment=False, rounds=1, seed=None):
    if self.time_stretching:
      pass

    if self.pitch_shifting:
      pass

    if self.dynamic_range_compression:
      pass

    if self.background_noise:
      pass


class Iterator(object):
  """Abstract base class for music data iterators.
  # Arguments
      n: Integer, total number of samples in the dataset to loop over.
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      seed: Random seeding for data shuffling.
  """
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

class NumpyArrayIterator(Iterator):
  """Iterator yielding data from a Numpy array.
  # Arguments
      x: Numpy array of input data.
      y: Numpy array of targets data.
      music_data_generator: Instance of `MusicDataGenerator`
          to use for random transformations and normalization.
      batch_size: Integer, size of a batch.
      shuffle: Boolean, whether to shuffle the data between epochs.
      seed: Random seed for data shuffling.
  """

  def __init__(self, x, y, music_data_generator,
               batch_size=32, shuffle=False, seed=None):
    if y is not None and len(x) != len(y):
        raise ValueError('X (Music tensor) and y (labels) '
                         'should have the same length. '
                         'Found: X.shape = %s, y.shape = %s' %
                         (np.asarray(x).shape, np.asarray(y).shape))
    self.music_data_generator = music_data_generator
    super(NumpyArrayIterator, self).__init__(x.shape[0], batch_size, shuffle, seed)

  def next(self):
      """For python 2.x.
      # Returns
          The next batch.
      """
      # Keeps under lock only the mechanism which advances
      # the indexing of each batch.
      with self.lock:
          index_array, current_index, current_batch_size = next(self.index_generator)
      # The transformation of images is not under thread lock
      # so it can be done in parallel
      batch_x = np.zeros(tuple([current_batch_size] + list(self.x.shape)[1:]), dtype=K.floatx())
      for i, j in enumerate(index_array):
          x = self.x[j]
          x = self.image_data_generator.random_transform(x.astype(K.floatx()))
          x = self.image_data_generator.standardize(x)
          batch_x[i] = x
      if self.save_to_dir:
          for i in range(current_batch_size):
              img = array_to_img(batch_x[i], self.data_format, scale=True)
              fname = '{prefix}_{index}_{hash}.{format}'.format(prefix=self.save_prefix,
                                                                index=current_index + i,
                                                                hash=np.random.randint(1e4),
                                                                format=self.save_format)
              img.save(os.path.join(self.save_to_dir, fname))
      if self.y is None:
          return batch_x
      batch_y = self.y[index_array]
      return batch_x, batch_y