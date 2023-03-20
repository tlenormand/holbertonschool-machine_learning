#!/usr/bin/env python3
"""
    Functions:
        mat_mul: function that performs matrix multiplication
"""


def mat_mul(mat1, mat2):
    """ function that performs matrix multiplication

    Arguments:
        mat1: matrix 1
        mat2: matrix 2

    Returns:
        new matrix multiplicated
    """
    if (not mat1 or not mat2 or
            not isinstance(mat1, list) or not isinstance(mat2, list)
            or not isinstance(mat1[0], list) or not isinstance(mat2[0], list)):
        return None

    return [[sum(a * b for a, b in zip(X_row, Y_col)) for Y_col in zip(*mat2)]
            for X_row in mat1]
