#!/usr/bin/env python3
""" One Hot """

import tensorflow.keras as K


def one_hot(labels, classes=None):
    """ Function that converts a label vector into a one-hot matrix

    Arguments:
        labels: label vector
        classes: classes for the one-hot matrix

    Returns:
        The one-hot matrix
    """
    return K.utils.to_categorical(labels, classes)
