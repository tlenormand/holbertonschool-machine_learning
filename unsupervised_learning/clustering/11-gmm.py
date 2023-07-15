#!/usr/bin/env python3
""" GMM """

import numpy as np

expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def gmm(X, k):
    """ Function that calculates a GMM from a dataset

    Arguments:
        X np.ndarray of shape (n, d) containing the dataset
        k positive integer containing the number of clusters

    Returns:
        pi, m, S, clss, bic
        pi np.ndarray shape (k,) containing the priors for each cluster
        m np.ndarray shape (k, d) containing the centroid means for each
            cluster
        S np.ndarray shape (k, d, d) containing the covariance matrices for
            each cluster
        clss np.ndarray shape (n,) containing the cluster indices for each
            data point
        bic np.ndarray shape (kmax - kmin + 1) containing the BIC value for
            each cluster size tested
    """
    if type(X) is not np.ndarray or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) is not int or k <= 0:
        return None, None, None, None, None

    n, d = X.shape
    pi = np.full((k,), 1 / k)
    m = np.random.uniform(np.min(X, axis=0), np.max(X, axis=0), size=(k, d))
    S = np.full((k, d, d), np.identity(d))
    clss = np.zeros((n,))
    l_prev = 0
    tol = 1e-5
    g, log = expectation(X, pi, m, S)
    while abs(log - l_prev) > tol:
        pi, m, S = maximization(X, g)
        g, log = expectation(X, pi, m, S)
        l_prev = log
    clss = np.argmax(g, axis=0)
    bic = None

    return pi, m, S, clss, bic
