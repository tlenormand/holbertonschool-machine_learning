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
    if (X.shape[1] != S.shape[0] or S.shape[0] != S.shape[1] or
            X.shape[1] != m.shape[0]):
        return None

    _, d = X.shape
    m = m.reshape((1, d))

    # calculate determinant of S
    det = np.linalg.det(S)

    P = 1 / np.sqrt(((2 * np.pi) ** d) * det) * np.exp(-1 / 2 * np.sum(
        np.matmul(np.linalg.inv(S), (X - m).T).T * (X - m), axis=1))

    # set all values in P that are less than 1e-300 to 1e-300
    np.place(P, P < 1e-300, 1e-300)

    return P
