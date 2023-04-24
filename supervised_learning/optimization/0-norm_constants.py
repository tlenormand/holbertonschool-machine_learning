#!/usr/bin/env python3
""" Normalization Constants """

import numpy as np


def normalization_constants(X):
    """ Calculates the normalization (standardization) constants of a matrix

    Argumets:
        X is the numpy.ndarray of shape (m, nx) to normalize

    Returns:
        the mean and standard deviation of each feature, respectively
    """
    return np.mean(X, axis=0), np.std(X, axis=0)
