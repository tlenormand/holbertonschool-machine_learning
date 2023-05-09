#!/usr/bin/env python3
""" Save and Load Weights """

import tensorflow.keras as K


def save_weights(network, filename, save_format='h5'):
    """ Function that saves a model's weights

    Arguments:
        network: model whose weights should be saved
        filename: path of the file that the weights should be saved to
        save_format: format in which the weights should be saved

    Returns:
        None
    """
    # h5 refers to the Hierarchical Data Format, a format designed to store
    # and organize large amounts of data
    # ref: https://www.h5py.org/
    network.save_weights(filename, save_format=save_format)


def load_weights(network, filename):
    """ Function that loads a model's weights

    Arguments:
        network: model to which the weights should be loaded
        filename: path of the file that the weights should be loaded from

    Returns:
        None
    """
    network.load_weights(filename)
