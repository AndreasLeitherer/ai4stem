import tensorflow
from tensorflow.keras.layers import Conv2D, LeakyReLU, MaxPool2D, Flatten, Dropout, Dense, Input

from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import ModelCheckpoint

from tensorflow.keras.models import load_model
from tensorflow.keras.models import Model

import numpy as np


from sklearn.metrics import accuracy_score

# from hyperopt import hp, fmin, tpe, STATUS_OK, Trials
# from hyperopt import space_eval

import os

import time
from tensorflow.keras.regularizers import L2


from scipy import stats

import logging

def predict_with_uncertainty(data, model, model_type='classification', n_iter=100):
    """
    This function allows to calculate the uncertainty of a neural network model using 
    Monte Carlo dropout. This follows Chap. 3 in Yarin Gal's PhD thesis:
    http://mlg.eng.cam.ac.uk/yarin/thesis/thesis.pdf
    We calculate the uncertainty of the neural network predictions in the three ways proposed in Gal's PhD thesis,
     as presented at pag. 51-54:
    - variation_ratio: defined in Eq. 3.19
    - predictive_entropy: defined in Eq. 3.20
    - mutual_information: defined at pag. 53 (no Eq. number)

    Parameters
    ----------
    data : ndarray
        Input data (note: has to be provided in a shape that 
        is compatible with the neural-network input shape).
    model : keras.engine.functional.Functional
        Keras/Tensorflow model object.
    model_type : string, optional
        Classification task, either 'regression' or 'classification'.
        The default is 'classification'.
    n_iter : int, optional
        Number of Monte Carlo samples. The default is 100.

    Raises
    ------
    ValueError
        If model_type is not 'regression' or 'classification',
        a ValueError is raised.

    Returns
    -------
    prediction : ndarray
        Numpy array containing the (mean) predictions (for classification:
        corresponds to the classification probabilities, averaged
        over the Monte Carlo samples).
    uncertainty : dict
        Dictionary with three keys whose corresponding 
        values are three uncertainty quantifiers (variation ratio,
        predictive entropy, and mutual information). These values 
        are 1D numpy arrays.
        
    .. codeauthors:: Angelo Ziletti <angelo.ziletti@gmail.com>, 
                    Andreas Leitherer <andreas.leitherer@gmail.com>

    """
    logging.info("Calculate predictions for {} images using {} Monte Carlo samples.".format(data.shape[0],
                                                                                            n_iter))
    labels = []
    results = []
    for idx_iter in range(n_iter):
        if (idx_iter % (int(n_iter) / 10 + 1)) == 0:
            logging.info("Performing forward pass: {0}/{1}".format(idx_iter + 1, n_iter))

        result = model.predict(data, verbose=0)
        label = result.argmax(axis=-1)

        labels.append(label)
        results.append(result)

    results = np.asarray(results)
    prediction = results.mean(axis=0)
    
    logging.info("Calculate uncertainty.")

    if model_type == 'regression':
        predictive_variance = results.var(axis=0)
        uncertainty = dict(predictive_variance=predictive_variance)

    elif model_type == 'classification':
        # variation ratio
        mode, mode_count = stats.mode(np.asarray(labels))
        variation_ratio = np.transpose(1. - mode_count.mean(axis=0) / float(n_iter))

        # predictive entropy
        # clip values to 1e-12 to avoid divergency in the log
        prediction = np.clip(prediction, a_min=1e-12, a_max=None, out=prediction)
        log_p_class = np.log2(prediction)
        entropy_all_iteration = - np.multiply(prediction, log_p_class)
        predictive_entropy = np.sum(entropy_all_iteration, axis=1)

        # mutual information
        # clip values to 1e-12 to avoid divergency in the log
        results = np.clip(results, a_min=1e-12, a_max=None, out=results)
        p_log_p_all = np.multiply(np.log2(results), results)
        exp_p_omega = np.sum(np.sum(p_log_p_all, axis=0), axis=1)
        mutual_information = predictive_entropy + 1. / float(n_iter) * exp_p_omega

        uncertainty = dict(variation_ratio=variation_ratio, predictive_entropy=predictive_entropy,
                           mutual_information=mutual_information)
    else:
        raise ValueError("Supported model types are 'classification' or 'regression'."
                         "model_type={} is not accepted.".format(model_type))

    return prediction, uncertainty



def decode_preds(data, model, n_iter=100):
    """
    Function for calculating mean predictions 
    (and no further uncertainty quantifier).

    Parameters
    ----------
    data : ndarray
        Input data (note: has to be provided in a shape that 
        is compatible with the neural-network input shape).
    model : keras.engine.functional.Functional
        Keras/Tensorflow model object.
    n_iter : int, optional
        Number of Monte Carlo samples. The default is 100.

    Returns
    -------
    predictions : TYPE
        DESCRIPTION.
        
    .. codeauthor:: Andreas Leitherer <andreas.leitherer@gmail.com>

    """

    results = []
    for idx in range(n_iter):
        pred = model.predict(data, verbose=0)
        results.append(pred)

    results = np.asarray(results)
    predictions = np.mean(results, axis=0)
    return predictions

def cnn_model(input_shape=(64, 64, 1), dropout=0.07, alpha=0.0, 
              nb_blocks = 3, filter_sizes=[32, 16, 8], kernel_size=(3,3), 
              nb_classes=10, l2_value=0.0):
    """
    Create Convolutional-Neural-Network classifier. Standard settings
    correspond to the model employed in Leitherer et al. arXiv:2303.12702 (2023).

    Parameters
    ----------
    input_shape : tuple, optional
        Model input shape. The default is (64, 64, 1).
    dropout : float, optional
        Dropout ratio applied to all weight layers. The default is 0.07.
    alpha : float, optional
        Parameter shifting activation origin in Leaky ReLU activation function. 
        The default is 0.0.
    nb_blocks : int, optional
        Number of layer blocks where one block contains
        two convolutional layers and one Max Pooling operation. The default is 3.
    filter_sizes : list, optional
        1D list containing the size of the convolutional filters
        in each block (same filter size used for both convolutional
        layers of a block). The default is [32, 16, 8].
    kernel_size : tuple, optional
        Kernel size used for all convolutional layers. The default is (3,3).
    nb_classes : int, optional
        Number of classes. The default is 10.
    l2_value : float, optional
        L2 regularization parameter. The default is 0.0.

    Raises
    ------
    ValueError
        Check that number of filters and the filter sizes have the same length.

    Returns
    -------
    model : keras.engine.functional.Functional
        TCompiled Tensorflow/Keras model.
    
    .. codeauthor:: Andreas Leitherer <andreas.leitherer@gmail.com>

    """

    if not len(filter_sizes) == nb_blocks:
        raise ValueError("# filters must be compatible with nb_blocks.")

    l2_reg = L2(l2=l2_value)

    inputs = Input(shape=input_shape, name='input')

    convlayer_counter = 0
    pooling_counter = 0
    activation_counter = 0
    dropout_counter = 0
    for i, filters in enumerate(filter_sizes):
        if i == 0:
            x = inputs

        for j in range(2): # repeat the following blocks 2 times

            x = Conv2D(filters=filters, kernel_size=kernel_size, strides=(1, 1),
                       padding='same', kernel_initializer='Orthogonal',
                       data_format="channels_last", kernel_regularizer=l2_reg,
                       name='Convolution_2D_{}'.format(convlayer_counter))(x)
            convlayer_counter += 1
            x = LeakyReLU(alpha=alpha, name='Leaky_ReLU_{}'.format(activation_counter))(x)
            activation_counter += 1
            x = Dropout(rate=dropout, name='Dropout_{}'.format(dropout_counter))(x, training=True)
            dropout_counter += 1

        if not i == (nb_blocks - 1):
            # for last block, no max pooling done
            x = MaxPool2D(pool_size=(2, 2), strides=(2, 2),
                          data_format="channels_last",
                          name='MaxPooling2D_{}'.format(pooling_counter))(x)
            pooling_counter += 1

    x = Flatten(name='Flatten')(x)
    x = Dense(128, name='Dense_1', kernel_regularizer=l2_reg, activation='relu')(x)
    x = Dropout(rate=dropout,
                name='dropout_{}'.format(dropout_counter))(x, training=True)
    outputs = Dense(nb_classes, name='Dense_2', activation='softmax') (x)

    model = tensorflow.keras.Model(inputs, outputs)

    adam = Adam(beta_1=0.9, beta_2=0.999, decay=0.0)

    model.compile(loss='categorical_crossentropy', optimizer=adam, metrics=['categorical_accuracy'])

    return model




def train_and_test_model(model, X_train, y_train, X_val, y_val, savepath_model,
                         epochs=10, batch_size=64, verbose=1, n_iter=100):
    """
    Conduct training and testing for given model and training/test data. 
    A classification task is assumed in this function.

    Parameters
    ----------
    model : keras.engine.functional.Functional
        Tensorflow/Keras model.
    X_train : ndarray
        Training data.
    y_train : array
        Training labels (one-hot encoded).
    X_val : ndarray
        Validation data.
    y_val : array
        Validation labels (one-hot encoded).
    savepath_model : string
        Path where to save the neural-network model.
    epochs : int, optional
        Number of training epochs. The default is 10.
    batch_size : int, optional
        Batch size employed in SGD training. The default is 64.
    verbose : int, optional
        Verbosity setting during training. The default is 1.
    n_iter : int, optional
        Number of Monte Carlo samples. The default is 100.

    Returns
    -------
    acc_train : float
        Accuracy on training set.
    acc_val : float
        Accuracy on validation set.
    optimal_model : keras.engine.functional.Functional
        Trained model (the one with optimal performance,
        here with minimal validation loss, which is the strategy 
        employed in Leitherer et al. arXiv:2303.12702 (2023)).
    history : dict
        History dictionary containing training/validation loss/accuracy
        over all training epochs.
        
    .. codeauthor:: Andreas Leitherer <andreas.leitherer@gmail.com>

    """

    callbacks_savepath = os.path.join(savepath_model, 'model_intermediate.h5')

    callbacks = []
    monitor = 'val_loss'
    mode = 'min'
    # Alternative: monitor validation accuracy.
    #monitor = 'val_categorical_accuracy'
    #mode = 'max'    
    save_model_per_epoch = ModelCheckpoint(callbacks_savepath, monitor=monitor, verbose=1,
                                       save_best_only=True, mode=mode, period=1)
    callbacks.append(save_model_per_epoch)

    # Additional possibility: use Early stopping.
    #es = EarlyStopping(monitor=monitor, mode=mode, patience=10)
    #callbacks.append(es)

    # Fit model
    logging.info('Start model fitting.')
    history = model.fit(X_train, y_train, epochs=epochs, batch_size=batch_size,
                        verbose=verbose, validation_data=(X_val, y_val),
                        callbacks=callbacks)
    logging.info('Model fitting finished, now load optimal model.')
    optimal_model = load_model(callbacks_savepath)

    logging.info('Calculate training predictions and classification accuracy.')
    train_pred = decode_preds(data=X_train, model=optimal_model, n_iter=n_iter)
    acc_train = accuracy_score(y_true=y_train.argmax(axis=-1), y_pred = train_pred.argmax(axis=-1))
    
    logging.info('Calculate validation predictions and classification accuracy.')
    val_pred = decode_preds(data=X_val, model=optimal_model, n_iter=n_iter)
    acc_val = accuracy_score(y_true=y_val.argmax(axis=-1), y_pred = val_pred.argmax(axis=-1))

    logging.info('Model evaluation finished.')

    return acc_train, acc_val, optimal_model, history




def start_training(X_train, X_val, y_train, y_val, 
                   savepath_model=None, params=None):
    """
    Given training data, perform training of Bayesian Convolutional
    Neural Network (standard settings reproduce model training in
    Leitherer et al. arXiv:2303.12702 (2023))

    Parameters
    ----------
    X_train : ndarray
        Training data.
    X_val : ndarray
        Validation data.
    y_train : array
        Training labels (one-hot-encoded format).
    y_val : TYPE
        Validation labels (one-hot-encoded format).
    savepath_model : string, optional
        Path where neural-network model is saved. The default is None and 
        in this case, the current path will be used.
    params : dict, optional
        Parameters specifying training procedure including 
        the number of epochs, batch size, alpha parameter (Leaky-ReLU parameter),
        kernel size, architecture (number of convolutional layers and filter 
        sizes per block), dropout ratio, L2 regularization parameter,
        and number of Monte Carlo samples. The default is None, in which case
        the settings in Leitherer et al. arXiv:2303.12702 (2023) are used.

    Returns
    -------
    acc_train : float
        Accuracy on training set.
    acc_val : float
        Accuracy on validation set.
    model : keras.engine.functional.Functional
        Trained model (the one with optimal performance,
        here with minimal validation loss, which is the strategy 
        employed in Leitherer et al. arXiv:2303.12702 (2023)).
    history : dict
        History dictionary containing training/validation loss/accuracy
        over all training epochs.
        
    .. codeauthor:: Andreas Leitherer <andreas.leitherer@gmail.com>

    """
    if savepath_model == None:
        savepath_model = os.getcwd()
    if params == None:
        params = {"epochs": 10, "batch_size": 64, "alpha": 0.0,
                "kernel_size": (3, 3),
                "architecture": (3, [32, 16, 8]),
                "dropout": 0.07,
                "l2_value": 0.0,
                'n_iter':10}
    
    logging.info('Start training.')
    t_start = time.time()
    
    logging.info('Define and compile Bayesian CNN model.')
    model = cnn_model(input_shape=(64, 64, 1), dropout=params["dropout"], alpha=params["alpha"],
                  nb_blocks=params["architecture"][0], filter_sizes=params["architecture"][1],
                  kernel_size=params["kernel_size"], nb_classes=np.unique(y_val.argmax(axis=-1)).size,
                  l2_value=params['l2_value'])

    acc_training, acc_validation, model, history = train_and_test_model(model=model, 
                                                                        batch_size=params['batch_size'],
                                                                        epochs=params['epochs'],
                                                                        X_train=X_train, 
                                                                        y_train=y_train,
                                                                        X_val=X_val, 
                                                                        y_val=y_val, 
                                                                        verbose=1,
                                                                        n_iter=params['n_iter'],
                                                                        savepath_model=savepath_model)
    t_end = time.time()
    eval_time = round(abs(t_start-t_end),3)
    logging.info('Training and model evaluation finished in {}s'.format(eval_time))
    
    return acc_training, acc_validation, model, history


def reshape_data_to_input_size(data, model):
    
    # reshape data
    input_shape_from_model = model.layers[0].get_input_at(0).get_shape().as_list()[1:]
    target_shape = tuple([-1] + input_shape_from_model)
    reshaped_data = np.reshape(data, target_shape)
    
    return reshaped_data

def get_truncated_model(model, layer_name):
    # Define model, where remove last classification layer
    inputs = model.input
    outputs = model.get_layer(layer_name).output
    truncated_model = Model(inputs=inputs,
                            outputs=outputs)
    return truncated_model

def get_nn_representations(model, data, 
                           layer_name='Dense_1', n_iter=100):
    
    truncated_model = get_truncated_model(model, layer_name)
    nn_representations = decode_preds(data, 
                                      truncated_model, 
                                      n_iter=n_iter)
    
    return nn_representations
