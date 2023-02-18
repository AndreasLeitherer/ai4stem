import tensorflow as tf
from tensorflow import keras

from keras.layers import Conv2D, LeakyReLU, MaxPool2D, Flatten, Dropout, Dense, Activation, Input
from keras.layers import BatchNormalization
from tensorflow.keras.optimizers import Adam

import numpy as np

from sklearn.model_selection import train_test_split
import json

if __name__ == '__main__':
    
    
    X_training = np.load('../data/fft_haadf_training_x.npy')
    y_training = np.load('../data/fft_haadf_training_y.npy')
    

    RANDOM_STATE = 42
    split_ratio = 0.2
    num_classes = 11
    
    # 80 /20 Split for creating training and test set 
    X_train, X_val, y_train, y_val = train_test_split(X_training, y_training,
                                            test_size=split_ratio, random_state=RANDOM_STATE,
                                            stratify=y_training)
    
    
    # convert class vectors to binary class matrices
    y_train = keras.utils.to_categorical(y_train, num_classes)
    y_val = keras.utils.to_categorical(y_val, num_classes)
    
    # reshape training and validation input
    X_train = np.moveaxis(X_train, 1, -1)
    X_val = np.moveaxis(X_val, 1, -1)
    #X_train = np.reshape(X_train, (X_train.shape[0], 64, 64, 3))
    #X_val = np.reshape(X_val, (X_val.shape[0], 64, 64, 3))
    #X_train = np.expand_dims(X_train, axis=-1)
    """
    X_train = np.asarray([X_train, X_train, X_train])
    X_train = np.reshape(X_train, (X_train.shape[0], 64, 64, 3))
    X_val = np.asarray([X_val, X_val, X_val])
    X_val = np.reshape(X_val, (X_val.shape[0], 64, 64, 3))
    """
    
    inputs = Input(shape=(64, 64, 3), name='input_1')
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='Orthogonal',
               data_format="channels_last", name='convolution2d_1')(inputs)
    x = LeakyReLU(alpha=0.1, name='leaky_re_lu_1')(x)
    x = Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='Orthogonal',
               data_format="channels_last", name='convolution2d_2')(x)
    x = LeakyReLU(alpha=0.1, name='leaky_re_lu_2')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                  data_format="channels_last", name='maxpooling2d_1')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same',  kernel_initializer='Orthogonal',
               data_format="channels_last", name='convolution2d_3')(x)
    x = LeakyReLU(alpha=0.1, name='leaky_re_lu_3')(x)
    x = Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='Orthogonal',
               data_format="channels_last", name='convolution_2d_4')(x)
    x = LeakyReLU(alpha=0.1, name='leaky_re_lu_4')(x)
    x = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                  data_format="channels_last", name='maxpooling_2d_2')(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='Orthogonal', 
               data_format="channels_last", name='convolution2d_5')(x)
    x = LeakyReLU(alpha=0.1, name='leaky_re_lu_5')(x)
    x = Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='same', kernel_initializer='Orthogonal',
               data_format="channels_last", name='convolution2d_6')(x)
    x = LeakyReLU(alpha=0.1, name='leaky_re_lu_6')(x)
    
    x = Flatten(name='flatten_1')(x)
    x = Dropout(rate=0.25, name='dropout_1')(x, training=True)
    x = Dense(128, name='dense_1')(x)
    x = BatchNormalization()(x)
    outputs = Dense(11, name='dense_2', activation='softmax') (x)
    #outputs = Activation(activation='softmax', name='activation_1')(x)
    
    model = keras.Model(inputs, outputs)
    
    adam = Adam(beta_1=0.9, beta_2=0.999, decay=0.0)
    
    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])
    
    # FIT
    params = {"epochs": 20, "batch_size": 64}
    history = model.fit(X_train, y_train,
                        validation_data = (X_val, y_val),
                        epochs=params["epochs"],
                        batch_size=params["batch_size"], verbose=True)
    # save new model
    model.save('../data/{}_cnn_model.h5'.format(model.name))
    with open('../data/{}_history.json'.format(model.name), 'w') as outfile:
        json.dump(history.history, outfile)