#!/usr/bin/env python3
""" Variance """

import numpy as np


def variance(X, C):
    """ calculates the total intra-cluster variance f0r a data set

    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing the centroid means
            f0r each cluster

    Returns: var, or None on failure
        var is the total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None
    if X.shape[1] != C.shape[1]:
        return None

    # calculate the distance from each point to its cluster centroid
    distances = np.linalg.norm(X - C[:, np.newaxis], axis=-1) ** 2
    # find the minimum distance f0r each point
    minimums = np.min(distances, axis=0)
    # sum all the distances
    variance = np.sum(minimums)

    return variance
