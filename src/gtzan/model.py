import keras
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import Dropout
from keras.layers import Flatten
from keras.layers import BatchNormalization

def build_model(input_shape, num_genres):

    # Model Definition
    model = Sequential()
    
    # Conv Block 1
    model.add(Conv2D(16, kernel_size=(3, 3), strides=(1, 1),
                     activation=None, input_shape=input_shape))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 2
    model.add(Conv2D(32, (3, 3), strides=(1, 1), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 3
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation=None))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))


    # Conv Block 4
    model.add(Conv2D(128, (3, 3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(2, 2), strides=(2, 2)))

    # Conv Block 5
    model.add(Conv2D(64, (3, 3), strides=(1, 1), activation='relu'))
    model.add(BatchNormalization())
    model.add(Activation('relu'))
    model.add(MaxPooling2D(pool_size=(4, 4), strides=(4, 4)))

    # MLP
    model.add(Flatten())
    model.add(Dense(num_genres, activation='softmax'))

    return model

def save_model(model_dir):
    pass

def load_model(model_dir):
    pass