#!/usr/bin/env python3
"""
    Functions:
        matrix_transpose: function that returns the transpose of a 2D matrix
"""


def matrix_transpose(matrix):
    """ function that returns the transpose of a 2D matrix

    Arguments:
        matrix: matrix to transpose

    Returns:
        transposed matrix
    """
    if not matrix or not isinstance(matrix, list):
        return []

    transposed = []

    for i in range(len(matrix[0])):
        transposed.append([])

    for i in range(len(matrix)):
        for j in range(len(matrix[0])):
            transposed[j].append(matrix[i][j])

    return transposed
