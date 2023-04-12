#!/usr/bin/env python3
""" Module that contains the function one_hot_encode """

import numpy as np


def one_hot_encode(Y, classes):
    """ Function that converts a numeric label vector into a one-hot matrix

    Arguments:
        Y: numpy.ndarray with shape (m,) containing numeric class labels
        classes: maximum number of classes found in Y

    Returns:
        A one-hot encoding of Y with shape (classes, m), or None on failure
    """
    if not isinstance(Y, np.ndarray) or len(Y) == 0:
        return None

    if not isinstance(classes, int) or classes < 2 or classes < Y.max():
        return None

    if np.sum(Y) == 0:
        return None

    array = np.zeros((classes, Y.size))

    for i in range(len(array[0])):
        array[Y[i]][i] = 1

    return array
