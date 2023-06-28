#!/usr/bin/env python3
""" Mean and Covariance """

import numpy as np


def mean_cov(X):
    """ calculates the mean and covariance of a data set

    Arguments:
        X (numpy.ndarray): data set of shape (n, d)

    Returns:
        mean, cov:
            mean is a numpy.ndarray of shape (1, d) containing the mean of the
                data set
            cov is a numpy.ndarray of shape (d, d) containing the covariance
                matrix of the data set
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        raise TypeError('X must be a 2D numpy.ndarray')

    n, _ = X.shape
    if n < 2:
        raise ValueError('X must contain multiple data points')

    mean = np.mean(X, axis=0, keepdims=True)
    cov = np.matmul((X - mean).T, (X - mean)) / (n - 1)

    return mean, cov
