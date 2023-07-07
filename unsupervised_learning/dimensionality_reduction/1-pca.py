#!/usr/bin/env python3
""" Dimensionality Reduction """

import numpy as np


def pca(X, ndim):
    """ performs PCA on a dataset

    Arguments:
        X {np.ndarray} -- dataset of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
        ndim {int} -- new dimensionality of the transformed X

    Returns:
        the transformed X of shape (n, ndim)
    """
    X = X - np.mean(X, axis=0)  # normalize X

    # do SVD on X to get the eigenvalues and eigenvectors
    u, s, vh = np.linalg.svd(X)

    # vh.T[:, :ndim] is the first ndim columns of vh transposed (ndim, d)
    w = vh.T[:, :ndim]

    # return the first ndim columns of vh
    return np.matmul(X, w)
