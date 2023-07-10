#!/usr/bin/env python3
""" Variance """

import numpy as np


def variance(X, C):
    """ calculates the total intra-cluster variance for a data set

    Args:
        X is a numpy.ndarray of shape (n, d) containing the data set
        C is a numpy.ndarray of shape (k, d) containing the centroid means
            for each cluster

    Returns: var, or None on failure
        var is the total variance
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(C, np.ndarray) or len(C.shape) != 2:
        return None

    variance = []
    for i in range(C.shape[0]):
        # calculate the distance from each point to its cluster centroid
        variance.append(np.linalg.norm(X - C[i], axis=1) ** 2)
    # find the minimum distance for each point
    variance = np.min(np.array(variance), axis=0)
    # sum all the distances
    variance = np.sum(variance)

    return variance
