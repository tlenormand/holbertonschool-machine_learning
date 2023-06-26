#!/usr/bin/env python3
""" Definiteness """

import numpy as np


def definiteness(matrix):
    """ calculates the definiteness of a matrix

    Arguments:
        matrix (list): matrix to calculate

    Returns:
        the definiteness of matrix
    """
    if type(matrix) is not np.ndarray:
        raise TypeError('matrix must be a numpy.ndarray')
    if len(matrix.shape) != 2 or matrix.shape[0] != matrix.shape[1]:
        return None
    if not np.all(np.transpose(matrix) == matrix):
        return None

    eigenvalues = np.linalg.eigvals(matrix)
    if all(eigenvalues > 0):
        return 'Positive definite'
    if all(eigenvalues >= 0):
        return 'Positive semi-definite'
    if all(eigenvalues < 0):
        return 'Negative definite'
    if all(eigenvalues <= 0):
        return 'Negative semi-definite'

    return 'Indefinite'
