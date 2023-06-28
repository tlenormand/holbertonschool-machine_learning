#!/usr/bin/env python3
""" Correlation """

import numpy as np


def correlation(C):
    """ calculates a correlation matrix

    Arguments:
        C (numpy.ndarray): covariance matrix of shape (d, d)

    Returns:
        correlation matrix of shape (d, d)
    """
    if type(C) is not np.ndarray:
        raise TypeError('C must be a numpy.ndarray')
    if len(C.shape) != 2 or C.shape[0] != C.shape[1]:
        raise ValueError('C must be a 2D square matrix')

    # C:
    # [[ 36 -30  15]
    #  [-30 100 -20]
    #  [ 15 -20  25]]

    diag = np.diag(C)  # shape = (d,)
    # [ 36 100  25]

    diag = np.expand_dims(diag, axis=0)  # shape = (1, d)
    # [[ 36 100  25]]

    sqrt = np.sqrt(diag)  # shape = (1, d)
    # [[ 6. 10.  5.]]

    sqrt = np.matmul(sqrt.T, sqrt)  # shape = (d, d)
    # [[ 36.  60.  30.]
    #  [ 60. 100.  50.]
    #  [ 30.  50.  25.]]

    corr = C / sqrt  # shape = (d, d)
    # [[ 1.  -0.5  0.5]
    #  [-0.5  1.  -0.4]
    #  [ 0.5 -0.4  1. ]]

    return corr
