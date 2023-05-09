#!/usr/bin/env python3
""" Train """

import tensorflow.keras as K


def train_model(network, data, labels, batch_size, epochs,
                validation_data=None, early_stopping=False,
                patience=0, learning_rate_decay=False, alpha=0.1,
                decay_rate=1, save_best=False, filepath=None,
                verbose=True, shuffle=False):
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
        decay_rate: the decay rate
        alpha: the initial learning rate
        learning_rate_decay: boolean that indicates whether learning rate decay
            should be used
        patience: patience used for early stopping
        early_stopping: boolean that indicates whether early stopping should be
            used
        validation_data: the data to validate the model with, if not None
        save_best: boolean indicating whether to save the model after each
            epoch if it is the best
        filepath: file path where the model should be saved

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

    if learning_rate_decay and validation_data:
        lr_decay = K.callbacks.LearningRateScheduler(
            schedule=alpha / (1 + decay_rate * epochs),
            verbose=1
        )
        callbacks.append(lr_decay)

    if save_best:
        checkpoint = K.callbacks.ModelCheckpoint(
            filepath=filepath,
            monitor='val_loss',
            save_best_only=True
        )
        callbacks.append(checkpoint)

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
