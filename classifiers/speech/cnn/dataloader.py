import json
import random
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score, f1_score, precision_score, recall_score
# keras
import keras

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