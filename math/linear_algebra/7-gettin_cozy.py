#!/usr/bin/env python3
"""
    Functions:
        cat_matrices2D: function that concatenates two matrices
            along a specific axis
"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def cat_matrices2D(mat1, mat2, axis=0):
    """ function that concatenates two matrices along a specific axis

    Arguments:
        mat1: first matrice
        mat2: second matrice
        axis: axis to concatenate

    Returns:
        new matrix concatenate
    """
    shape_mat1 = matrix_shape(mat1)
    shape_mat2 = matrix_shape(mat2)

    if len(shape_mat1) != len(shape_mat2):
        return None

    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
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
