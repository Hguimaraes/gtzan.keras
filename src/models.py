import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization

"""
"""
def cnn_gtzan_model(input_shape):
  model = Sequential()
  
  # First Conv Layer
  model.add(Conv1D(256, 4, input_shape = input_shape))
  model.add(keras.layers.advanced_activations.LeakyReLU())
  model.add(BatchNormalization())
  model.add(MaxPooling1D(pool_size=4, strides=4))
  
  # Second Conv Layer
  model.add(Conv1D(384, 4))
  model.add(keras.layers.advanced_activations.LeakyReLU())
  model.add(BatchNormalization())
  model.add(GlobalMaxPooling1D())

  # Regular MLP
  model.add(Dense(512))
  model.add(keras.layers.advanced_activations.LeakyReLU())
  model.add(Dropout(0.25))
  model.add(Dense(10, activation='softmax'))

  return model