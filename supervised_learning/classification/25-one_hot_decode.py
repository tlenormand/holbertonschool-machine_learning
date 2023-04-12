#!/usr/bin/env python3
""" Module that contains the function one_hot_decode """

import numpy as np


def one_hot_decode(one_hot):
    """ Function that converts a one-hot matrix into a vector of labels

    Arguments:
        one_hot: one-hot encoded numpy.ndarray with shape (classes, m)

    Returns:
        A numpy.ndarray with shape (m, ) containing the numeric labels for each
        example, or None on failure
    """
    if not isinstance(one_hot, np.ndarray) or len(one_hot) == 0:
        return None
    
    x, y = one_hot.shape
    if x < 2 or y < 2:
        return None

    # argmax returns the indices of the maximum values along an axis
    return np.argmax(one_hot, axis=0)
