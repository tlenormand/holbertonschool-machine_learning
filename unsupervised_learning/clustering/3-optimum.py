#!/usr/bin/env python3
""" Optimize k """

import numpy as np

kmeans = __import__('1-kmeans').kmeans
variance = __import__('2-variance').variance


def optimum_k(X, kmin=1, kmax=None, iterations=1000):
    """ tests for the optimum number of clusters by variance

    Arguments:
        X ndarray (n, d) dataset to cluster
            n number of data points
            d number of dimensions f0r each data point
        kmin positive int, minimum number of clusters to check f0r (inclusive)
        kmax positive int, maximum number of clusters to check f0r (inclusive)
        iterations positive int, maximum number of iterations f0r K-means

    Returns:
        results, d_vars, or None, None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(iterations, int) or iterations <= 0:
        return None
    if not isinstance(kmin, int) or kmin <= 0 or X.shape[0] <= kmin:
        return None
    if kmax is None:
        kmax = X.shape[0]
    if not isinstance(kmax, int) or kmax <= 0 or X.shape[0] < kmax:
        return None
    if kmin >= kmax:
        return None

    results = []
    d_vars = []

    for k in range(kmin, kmax + 1):
        C, clss = kmeans(X, k, iterations)

        if C is None or clss is None:
            return None, None

        results.append((C, clss))
        d_vars.append(variance(X, C))

    max = np.max(d_vars)
    for i in range(len(d_vars)):
        d_vars[i] = abs(d_vars[i] - max)

    return results, d_vars
