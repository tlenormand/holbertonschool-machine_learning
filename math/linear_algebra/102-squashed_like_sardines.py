#!/usr/bin/env python3
"""
    Functions:
        cat_matrices: function that concatenates two matrices along a specific axis
"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def cat_matrices(mat1, mat2, axis=0):
    """ function that concatenates two matrices along a specific axis

    Arguments:
        mat1: first matrix
        mat2: second matrix
        axis: axis to concatenate

    Returns:
        new matrix with the concatenation of each element
    """
    if axis == 0:
        if len(mat1) != len(mat2):
            return None
        else:
            return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        else:
            return [
                mat1[i] + mat2[i] for i in range(len(mat1))
            ]
