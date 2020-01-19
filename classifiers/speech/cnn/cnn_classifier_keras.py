import numpy as np
# Keras
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D,BatchNormalization, MaxPooling2D, Dense, Activation, Reshape, Dropout
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras import optimizers
# config
from classifiers.speech.cnn.config import model_config as config

class CNNClassifier():
    def __init__(self, input_shape):
        self.config = config['cnn']
        model = Sequential(name=self.config['model_name'])

        # LFLB1
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same', data_format='channels_last', input_shape=input_shape))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.5))

        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))

        model.add(Conv2D(filters=32, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.5))

        '''
        model.add(Conv2D(filters=64, kernel_size=(3,3), strides=(1,1), padding='same'))
        model.add(BatchNormalization())
        model.add(Activation('elu'))
        model.add(MaxPooling2D(pool_size=(2,2), strides=(2,2)))
        model.add(Dropout(0.25))
        '''

        model.add(Flatten())
        # FC
        model.add(Dense(units=self.config['num_classes'], activation='softmax'))

        # Model compilation
        opt = optimizers.Adam(lr=self.config['learning_rate_max'], beta_1=0.9,  beta_2=0.999, amsgrad=False)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model = model

    def load(self):
        return self.model
