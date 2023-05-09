#!/usr/bin/env python3
""" model """

import tensorflow.keras as K


def save_model(network, filename):
    """ Function that saves an entire model

    Arguments:
        network: model to save
        filename: path of the file that the model should be saved to

    Returns:
        None
    """
    network.save(filename)


def load_model(filename):
    """ Function that loads an entire model

    Arguments:
        filename: path of the file that the model should be loaded from

    Returns:
        the loaded model
    """
    return K.models.load_model(filename)
