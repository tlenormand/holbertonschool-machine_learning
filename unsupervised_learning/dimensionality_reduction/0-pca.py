#!/usr/bin/env python3
""" Dimensionality Reduction """

import numpy as np


def pca(X, var=0.95):
    """ performs PCA on a dataset

    Arguments:
        X {np.ndarray} -- dataset of shape (n, d) where:
            n is the number of data points
            d is the number of dimensions in each point
        var {float} -- fraction of the variance that the PCA transformation
            should maintain

    Returns:
        the weights matrix, W, that maintains var fraction of X's original
        variance
    """
    # do SVD on X to get the eigenvalues and eigenvectors
    u, s, vh = np.linalg.svd(X)

    # calculate the cumulative sum of the eigenvalues
    cum_var = np.cumsum(s) / np.sum(s)

    # get the index of the first value where the cumulative sum is >= var
    r = np.argwhere(cum_var >= var)[0, 0]

    # return the first r + 1 columns of vh
    return vh[:r + 1].T
