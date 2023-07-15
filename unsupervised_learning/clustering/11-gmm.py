#!/usr/bin/env python3
""" GMM """

import numpy as np
import sklearn.mixture


def gmm(X, k):
    """ calculates a GMM from a dataset

    Arguments:
        X np.ndarray (n, d) data set
        k positive int number of clusters

    Returns:
        pi, m, S, clss, bic
        pi np.ndarray (k,) priors for each cluster
        m np.ndarray (k, d) centroid means for each cluster
        S np.ndarray (k, d, d) covariance matrices for each cluster
        clss np.ndarray (n,) cluster indices for data points
        bic np.ndarray (kmax - kmin + 1,) BIC value for each cluster size
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if not isinstance(k, int) or k <= 0:
        return None, None, None, None, None

    gmm = sklearn.mixture.GaussianMixture(n_components=k).fit(X)
    pi = gmm.weights_
    m = gmm.means_
    S = gmm.covariances_
    clss = gmm.predict(X)
    bic = gmm.bic(X)

    return pi, m, S, clss, bic
