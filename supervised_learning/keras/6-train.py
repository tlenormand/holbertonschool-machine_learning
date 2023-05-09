#!/usr/bin/env python3
""" Early Stopping """

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, verbose=True, shuffle=False):
    """ Function that trains a model using mini-batch gradient descent

    Arguments:
        network: model to train
        data: numpy.ndarray of shape (m, nx) containing the input data
        labels: one-hot numpy.ndarray of shape (m, classes) containing
            the labels of data
        batch_size: size of the batch used for mini-batch gradient descent
        epochs: number of passes through data for mini-batch gradient descent
        verbose: boolean that determines if output should be printed during
            training
        shuffle: boolean that determines whether to shuffle the batches every
            epoch

    Returns:
        The History object generated after training the model
    """
    # callbacks refers to a list of callbacks to apply during training
    # ref: https://keras.io/api/models/model_training_apis/#fit-method
    callbacks = []

    if early_stopping and validation_data:
        early_stopping = K.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience
        )
        callbacks.append(early_stopping)

    return network.fit(
        x=data,
        y=labels,
        batch_size=batch_size,
        epochs=epochs,
        verbose=verbose,
        shuffle=shuffle,
        validation_data=validation_data,
        callbacks=callbacks
    )
