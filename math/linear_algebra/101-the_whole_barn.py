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
    if (not mat1 or not mat2 or
            matrix_shape(mat1) != matrix_shape(mat2)):
        return None

    if isinstance(mat1, (int, float)):
        return mat1 + mat2

    result = []

    # print("mat1: ", mat1, len(mat1))
    for i in range(len(mat1)):
        if isinstance(mat1, list):
            result.append(add_matrices(mat1[i], mat2[i]))

    return result
