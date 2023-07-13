#!/usr/bin/env python3
""" Maximization """

import numpy as np


def maximization(X, g):
    """ calculates the maximization step in the EM algorithm for a GMM

    Arguments:
        X np.ndarray (n, d) data set
        g np.ndarray (k, n) posterior probabilities for each data point
            in each cluster
        k number of clusters

    Returns:
        pi, m, S, or None, None, None on failure
        pi np.ndarray (k,) updated priors for each cluster
        m np.ndarray (k, d) updated centroid means for each cluster
        S np.ndarray (k, d, d) updated covariance matrices for each cluster
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None
    if not isinstance(g, np.ndarray) or len(g.shape) != 2:
        return None, None, None
    if X.shape[0] != g.shape[1]:
        return None, None, None
    if not np.isclose(np.sum(g, axis=0), np.ones((X.shape[0],))).all():
        return None, None, None

    n, d = X.shape
    k = g.shape[0]

    # initialize pi, m, S
    pi = np.zeros((k,))
    m = np.zeros((k, d))
    S = np.zeros((k, d, d))

    for i in range(k):
        # calculate pi, m, S
        pi[i] = np.sum(g[i]) / n
        m[i] = np.matmul(g[i], X) / np.sum(g[i])
        S[i] = np.matmul(g[i] * (X - m[i]).T, (X - m[i])) / np.sum(g[i])

    return pi, m, S
