#!/usr/bin/env python3
""" Initialize K-means """

import numpy as np


def initialize(X, k):
    """ initializes cluster centroids for K-means

    Arguments:
        X ndarray (n, d) dataset to cluster
            n number of data points
            d number of dimensions for each data point
        k positive int, number of clusters

    Returns: ndarray (k, d) initialized centroids
        or None on failure
    """
    return initialize(X, k-1) if k < 1 else np.random.uniform(
            np.min(X, axis=0),
            np.max(X, axis=0),
            (k, X.shape[1])
        )
