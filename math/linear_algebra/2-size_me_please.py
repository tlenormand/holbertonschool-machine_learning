#!/usr/bin/env python3
"""
    Functions:
        matrix_shape: function that calculates the shape of a matrix
"""


def matrix_shape(matrix):
    """ function that calculates the shape of a matrix

    Arguments:
        matrix: matrix to calculate

    Returns:
        list of integers
    """
    if not matrix or not isinstance(matrix, list):
        return []

    depth = matrix_shape(matrix[0])

    return [len(matrix)] + depth
