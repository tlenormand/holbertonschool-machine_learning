#!/usr/bin/env python3
""" Expectation """

import numpy as np

pdf = __import__('5-pdf').pdf


def expectation(X, pi, m, S):
    """ calculates the expectation step in the EM algorithm for a GMM

    Arguments:
        X np.ndarray (n, d) data set
        pi np.ndarray (k,) priors for each cluster
        m np.ndarray (k, d) centroid means for each cluster
        S np.ndarray (k, d, d) covariance matrices for each cluster

    Returns:
        g, l, or None, None on failure
        g np.ndarray (k, n) posterior probabilities for each data point
        l total log likelihood
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(pi, np.ndarray) or len(pi.shape) != 1:
        return None, None
    if not isinstance(m, np.ndarray) or len(m.shape) != 2:
        return None, None
    if type(pi) is not np.ndarray or len(pi.shape) != 1:
        return None, None
    if not isinstance(S, np.ndarray) or len(S.shape) != 3:
        return None, None
    if (X.shape[1] != S.shape[1] or S.shape[1] != S.shape[2] or
            X.shape[1] != m.shape[1] or pi.shape[0] != m.shape[0] or
            pi.shape[0] != S.shape[0]):
        return None, None

    # number of data points
    n = X.shape[0]
    # number of clusters
    k = pi.shape[0]

    # calculate g using pdf function
    g = np.zeros((k, n))
    for i in range(k):
        g[i] = pi[i] * pdf(X, m[i], S[i])

    # calculate l using log likelihood
    log_l = np.sum(np.log(np.sum(g, axis=0)))
    # normalize g
    g = g / np.sum(g, axis=0)

    return g, log_l
