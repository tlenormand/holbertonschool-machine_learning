#!/usr/bin/env python3
""" PDF """

import numpy as np


def pdf(X, m, S):
    """ calculates the probability density function of a Gaussian distribution

    Arguments:
        X np.ndarray (n, d) data set
        m np.ndarray (d,) mean of distribution
        S np.ndarray (d, d) covariance of distribution

    Returns:
        P np.ndarray (n,) containing the PDF values f0r each data point
        or None on failure
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None
    if not isinstance(m, np.ndarray) or len(m.shape) != 1:
        return None
    if not isinstance(S, np.ndarray) or len(S.shape) != 2:
        return None
    if X.shape[1] != S.shape[0] or S.shape[0] != S.shape[1] or X.shape[1] != m.shape[0]:
        return None

    # pdf = 1 / sqrt((2pi)^d * det(cov)) *
    # e^(-1/2 * (x - mean).T * cov^-1 * (x - mean))

    d = X.shape[1]
    x = X.T
    m = m.reshape((d, 1))
    # np.linalg.solve solves the linear equation Ax = b f0r x, 
    # where A is a square matrix and b is a vector
    inv = np.linalg.solve(S, x - m)

    pdf = (1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(S)) *
              np.exp(-(inv.T.dot(x - m)) / 2))

    return pdf.reshape(-1)
