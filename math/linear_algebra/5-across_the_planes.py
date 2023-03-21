#!/usr/bin/env python3
"""
    Functions:
        add_matrices2D: function that adds two matrices element-wise
        matrix_shape: function that calculates the shape of a matrix
"""


def add_matrices2D(mat1, mat2):
    """ function that adds two matrices element-wise

    Arguments:
        mat1: first matrice
        mat2: second matrice

    Returns:
        new matrice with the sum of each element
    """
    if (not mat1 or not mat2 or
            matrix_shape(mat1) != matrix_shape(mat2)):
        return None

    return [
        [mat1[i][j] + mat2[i][j] for j in range(len(mat1))]
        for i in range(len(mat1[0]))
    ]


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
