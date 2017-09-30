import keras
from keras.models import Sequential
from keras.models import Model
from keras.layers import Input
from keras.layers import Dense
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import Activation
from keras.layers.merge import concatenate
from keras.layers.convolutional import Conv1D
from keras.layers.pooling import MaxPooling1D
from keras.layers.pooling import GlobalMaxPooling1D
from keras.layers.pooling import GlobalAveragePooling1D
from keras.layers.normalization import BatchNormalization
from keras.layers.advanced_activations import LeakyReLU

# @Class: ModelZoo
# @Description: Set of models to use to solve the classification problem.
class ModelZoo(object):
  # @Method: cnn_melspect
  # @Description: 
  #  Method used for classify data from GTZAN in the 
  # MelSpectrogram input format.
  @staticmethod
  def cnn_melspect_1D(input_shape):
    kernel_size = 3
    #activation_func = LeakyReLU()
    activation_func = Activation('relu')
    inputs = Input(input_shape)

    # Convolutional block_1
    conv1 = Conv1D(32, kernel_size)(inputs)
    act1 = activation_func(conv1)
    bn1 = BatchNormalization()(act1)
    pool1 = MaxPooling1D(pool_size=2, strides=2)(bn1)

    # Convolutional block_2
    conv2 = Conv1D(64, kernel_size)(pool1)
    act2 = activation_func(conv2)
    bn2 = BatchNormalization()(act2)
    pool2 = MaxPooling1D(pool_size=2, strides=2)(bn2)

    # Convolutional block_3
    conv3 = Conv1D(128, kernel_size)(pool2)
    act3 = activation_func(conv3)
    bn3 = BatchNormalization()(act3)
    
    # Global Layers
    gmaxpl = GlobalMaxPooling1D()(bn3)
    gmeanpl = GlobalAveragePooling1D()(bn3)
    mergedlayer = concatenate([gmaxpl, gmeanpl], axis=1)

    # Regular MLP
    dense1 = Dense(512,
        kernel_initializer='glorot_normal',
        bias_initializer='glorot_normal')(mergedlayer)
    actmlp = activation_func(dense1)
    reg = Dropout(0.5)(actmlp)

    dense2 = Dense(512,
        kernel_initializer='glorot_normal',
        bias_initializer='glorot_normal')(reg)
    actmlp = activation_func(dense2)
    reg = Dropout(0.5)(actmlp)
    
    dense2 = Dense(10, activation='softmax')(reg)

    model = Model(inputs=[inputs], outputs=[dense2])
    return model