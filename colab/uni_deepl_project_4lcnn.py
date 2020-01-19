# To add a new cell, type '# %%'
# To add a new markdown cell, type '# %% [markdown]'
# %%
from IPython import get_ipython

# %%
#@markdown #SpeechToEmotion-keras
#@markdown ---
#@markdown using a 4 Layer 2DCNN to predict the 
#@markdown emotions out of Log-Mel-Spectrum features


# %%
#@markdown connect the google drive to colab notebook
from google.colab import drive
drive.mount('/content/gdrive')


# %%
#@title Imports
#@markdown import all important dependencies

get_ipython().run_line_magic('matplotlib', 'inline')
get_ipython().run_line_magic('reload_ext', 'autoreload')
get_ipython().run_line_magic('autoreload', '2')

import json
import random
import numpy as np
import matplotlib.pyplot as plt

# Keras
import keras
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau
from keras.models import Sequential
from keras.layers import Conv2D,BatchNormalization, MaxPooling2D, Dense, Activation, Reshape, Dropout
from keras.layers import Dense, Conv2D, Flatten
from keras.layers import Input, Flatten, Dropout, Activation, BatchNormalization
from keras import optimizers
from keras.models import load_model


# %%
model_config = {
    'cnn' : {
        'batch_size' : 64,
        'num_classes' : 4,
        'num_epochs' : 300,
        'padding_size' : 300,
        'learning_rate_min' : 0.000001,
        'learning_rate_max' : 0.001,
        'model_name': 'Audio_2DCNN_4L.h5'
    }
}

config = model_config['cnn']


# %%
#@title Dataloader
#@markdown create a dataloader do import the input data and export the predictions

class KerasDataloader():
    """
    Keras Dataloader
    The data must be a json file with the following structure:
    { "0" : {"features" : [[]] ,
            "activation" : 0|1 ,
            "valence" : 0|1},
      "1" : ...}
    }

    Parameters
    ------
    filepath : str
        path to the file location
    filename : str
        filename for the dataimport
    traindata: bool
        If True, the data will be split into train and test dataset with data and label lists.
        if False, just a datalist will be created
    """

    def __init__(self, filepath, filename, paddingsize, traindata=True):
        self.filepath = filepath
        self.filename = filename
        self.traindata = traindata
        self.padding_size = paddingsize
        self.lookup = { (0,0): [1.0,0.0,0.0,0.0],
                        (1,0): [0.0,1.0,0.0,0.0],
                        (0,1): [0.0,0.0,1.0,0.0],
                        (1,1): [0.0,0.0,0.0,1.0]
                      }
        self.data=[]
        self.__load__()

    def __load__(self):
        data = {}
        with open(self.filepath+self.filename) as jsonFile:
            data = json.load(jsonFile)
        for item in data:
            if self.traindata:
                self.data.append({'features': data[item]['features'], 
                                  'label': self.__onehot__(data[item]['valence'], data[item]['activation']) })
            else:
              self.data.append({'features': data[item]['features']})
    
    def __onehot__(self, valence, activation):
        return self.lookup[(valence, activation)]
        
    def __onehot_rev__(self, value):
        for label, onehot in self.lookup.items():
            if np.argmax(onehot) == np.argmax(value):
                return label[0], label[1]

    def load_data(self, splitvalue=0.1):
        """
        Function to load the input dataset

        Parameters
        ----------
        splitvalue : float
            split value between 0 and 1. Represents the percentage of the test, train set ratio
          
        Return
        ------
        Tupel: Lists
            (x_train, y_train), (x_val, y_val)
        
        """
        if self.traindata: 
            random.shuffle(self.data)
        x_train = []
        y_train = []
        x_val = []
        y_val = []
        if self.traindata:
            splitindex = int(len(self.data)*(1-splitvalue))
            train_set, val_set = self.data[:splitindex], self.data[splitindex+1:]
            for item in train_set:
                item['features'] = keras.preprocessing.sequence.pad_sequences(np.asarray(item['features']).transpose(), self.padding_size, padding='pre', value=0).transpose()
                x_train.append(item['features'])
                y_train.append(item['label'])
            for item in val_set:
                item['features'] = keras.preprocessing.sequence.pad_sequences(np.asarray(item['features']).transpose(), self.padding_size, padding='pre', value=0).transpose()
                x_val.append(item['features'])
                y_val.append(item['label'])
            
        else:
            for item in self.data:
                item['features'] = keras.preprocessing.sequence.pad_sequences(np.asarray(item['features']).transpose(), self.padding_size, padding='pre', value=0).transpose()
                x_train.append(item['features'])
        x_train=np.asarray(x_train)
        y_train=np.asarray(y_train)
        x_val=np.asarray(x_val)
        y_val=np.asarray(y_val)
        return (x_train, y_train), (x_val, y_val)

    def save_predictions(self, predictions):
        """
        Function to save the model prediction in json file with structure:
        { "0" : {"features" : [[]]},
          "1" : ...}
        }  
    
        Parameters
        ----------
        prediction : list
            a list of the predicted classes
          
        """
        result = {}
        for idx, prediction in enumerate(predictions):
            valence, activation = self.__onehot_rev__(prediction)
            result[str(idx)] = {'valence' : valence, 'activation': activation}
        with open(self.filepath + 'results.json', 'w') as fp:
            json.dump(result, fp)


# %%
dataloader = KerasDataloader('./gdrive/My Drive/develop/uni/deepl_project/data/','train.json', config['padding_size'])
(x_train, y_train), (x_val, y_val) = dataloader.load_data()
x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
#y_train = keras.utils.to_categorical(y_train, num_classes)
#y_val = keras.utils.to_categorical(y_val, num_classes)
target_shape_train = y_train.shape
target_shapte_test = y_val.shape 

input_shape_total = x_train.shape
input_shape = x_train.shape[1:]


# %%
#@title Create Model

class CNNClassifier():
    def __init__(self):
        model = Sequential(name=config['model_name'])

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
        model.add(Dense(units=config['num_classes'], activation='softmax'))

        # Model compilation
        opt = optimizers.Adam(lr=config['learning_rate_max'], beta_1=0.9,  beta_2=0.999, amsgrad=False)
        model.compile(optimizer=opt, loss='categorical_crossentropy', metrics=['categorical_accuracy'])
        self.model = model

    def load(self):
        return self.model


# %%
#load model
model = CNNClassifier().load()

# Summary
model.summary()

# Model Training
lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=config['learning_rate_min'])
# Please change the model name accordingly.
# Name format datatype_sequencelength_featurelength_architecture_layercount_epochs_accuracy:
mcp_save = ModelCheckpoint('./gdrive/My Drive/develop/uni/deepl_project/runs/models/'+ config['model_name'], save_best_only=True, monitor='val_categorical_accuracy', mode='max')
cnnhistory=model.fit(x_train, y_train, batch_size=config['batch_size'], epochs=config['num_epochs'],validation_data=(x_val, y_val), callbacks=[mcp_save, lr_reduce])


# %%
max(cnnhistory.history['val_categorical_accuracy'])


# %%
# Plotting the Train Valid Loss Graph

plt.plot(cnnhistory.history['categorical_accuracy'])
plt.plot(cnnhistory.history['val_categorical_accuracy'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


# %%
# Plotting the Train Valid Loss Graph

plt.plot(cnnhistory.history['loss'])
plt.plot(cnnhistory.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

# %%

dataloader_dev = KerasDataloader('./gdrive/My Drive/develop/uni/deepl_project/data/','dev.json', config['padding_size'], False)
(x_dev, y_train), (x_val, y_val) = dataloader_dev.load_data()
x_dev = x_dev.reshape((x_dev.shape[0], x_dev.shape[1], x_dev.shape[2], 1))

predictions = model.predict(x_dev)
print(predictions.shape)

# %%
dataloader_dev.save_predictions(predictions)

