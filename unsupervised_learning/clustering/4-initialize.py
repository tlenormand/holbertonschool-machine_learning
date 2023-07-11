#!/usr/bin/env python3
""" Initialize GMM """

import numpy as np

kmeans = __import__('1-kmeans').kmeans


def initialize(X, k):
    """ initializes variables for a Gaussian Mixture Model

    Arguments:
        X np.ndarray (n, d) dataset that will be used for K-means clustering
            n number of data points
            d number of dimensions for each data point
        k positive int containing the number of clusters

    Returns:
        pi np.ndarray (k,) containing the priors for each cluster, initialized
            evenly
        m np.ndarray (k, d) containing the centroid means for each cluster,
            initialized with K-means
        S np.ndarray (k, d, d) containing the covariance matrices for each
            cluster, initialized as identity matrices
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(k, int) or k <= 0 or X.shape[0] < k:
        return None, None, None

    # calculate pi
    pi = np.ones(k) / k
    # calculate m using previous kmeans function
    m, _ = kmeans(X, k)
    d = X.shape[1]
    # calculate S using identity matrix
    # np.tile repeats the identity matrix k times
    S = np.tile(np.identity(d), (k, 1)).reshape((k, d, d))

    return pi, m, S
