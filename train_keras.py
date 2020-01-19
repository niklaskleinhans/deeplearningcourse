import numpy as np
import matplotlib.pyplot as plt
from datetime import datetime

from classifiers.speech.cnn.cnn_classifier_keras import CNNClassifier
from classifiers.speech.cnn.dataloader import KerasDataloader
from classifiers.speech.cnn.config import model_config as configuration
from keras.callbacks import ModelCheckpoint, ReduceLROnPlateau

MODEL = 'CNN'


def train_cnn(config):
    # load data
    dataloader = KerasDataloader('./data/speech/','train.json', config['padding_size'])
    (x_train, y_train), (x_val, y_val) = dataloader.load_data()
    x_train = x_train.reshape((x_train.shape[0], x_train.shape[1], x_train.shape[2], 1))
    x_val = np.reshape(x_val, (x_val.shape[0], x_val.shape[1], x_val.shape[2], 1))
    input_shape = x_train.shape[1:]

    #load Model
    model = CNNClassifier(input_shape).load()    

    # set learners
    lr_reduce = ReduceLROnPlateau(monitor='val_loss', factor=0.9, patience=20, min_lr=config['learning_rate_min'])
    mcp_save = ModelCheckpoint('./runs/models/'+ config['model_name'], save_best_only=True, monitor='val_categorical_accuracy', mode='max')

    # train
    history = model.fit(x_train, y_train, batch_size=config['batch_size'], epochs=config['num_epochs'],validation_data=(x_val, y_val), callbacks=[mcp_save, lr_reduce])

    return model, history

def store_results(config, history):
    plt.subplot(2,1,1)
    plt.plot(history.history['categorical_accuracy'])
    plt.plot(history.history['val_categorical_accuracy'])
    plt.title('model accuracy')
    plt.ylabel('accuracy')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.subplot(2,1,2)
    plt.plot(history.history['val_loss'])
    plt.title('model loss')
    plt.ylabel('loss')
    plt.xlabel('epoch')
    plt.legend(['train', 'test'], loc='upper left')

    plt.savefig('./runs/results/' + config['model_name'] + '.png')

def predict(model, config):
    dataloader_dev = KerasDataloader('./data/speech/','dev.json', config['padding_size'], False)
    (x_dev, _),_ = dataloader_dev.load_data()
    x_dev = x_dev.reshape((x_dev.shape[0], x_dev.shape[1], x_dev.shape[2], 1))

    predictions = model.predict(x_dev)
    print(predictions.shape)

    dataloader_dev.save_predictions(predictions)

def train():
    # train
    if MODEL == 'CNN':
        config = configuration['cnn']
        config['timestamp'] = datetime.now().strftime("%m%d%Y-%H%M%S")
        model, history = train_cnn(config)
        print('BEST ACCURACY: ', max(history.history['val_categorical_accuracy']))

    # show the results
    store_results(config, history)

    #predict on validation dataset
    predict(model, config)

if __name__ == '__main__':
    train()