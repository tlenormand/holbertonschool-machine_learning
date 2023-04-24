#!/usr/bin/env python3
""" Normalization Constants """

import numpy as np


def normalize(X, m, s):
    """ Normalizes (standardizes) a matrix

    Argumets:
        X is the numpy.ndarray of shape (m, nx) to normalize
        m is a numpy.ndarray of shape (nx,) that contains the mean of all
            features of X
        s is a numpy.ndarray of shape (nx,) that contains the standard
            deviation of all features of X

    Returns:
        The normalized X matrix
    """
    return (X - m) / s
