#!/usr/bin/env python3
"""
    Functions:
        add_matrices: function that adds two matrices
"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def add_matrices(mat1, mat2):
    """ function that adds two matrices

    Arguments:
        mat1: first matrix
        mat2: second matrix

    Returns:
        new matrix with the sum of each element
    """
    if matrix_shape(mat1) != matrix_shape(mat2):
        return None

    if not isinstance(mat1, (list)):
        return mat1 + mat2

    return [
        add_matrices(mat1[i], mat2[i]) for i in range(len(mat1))
    ]
