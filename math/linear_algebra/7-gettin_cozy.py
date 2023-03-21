#!/usr/bin/env python3
"""
    Functions:
        cat_matrices2D: function that concatenates two matrices
            along a specific axis
"""


def cat_matrices2D(mat1, mat2, axis=0):
    """ function that concatenates two matrices along a specific axis

    Arguments:
        mat1: first matrice
        mat2: second matrice
        axis: axis to concatenate

    Returns:
        new matrix concatenate
    """
    if axis == 0:
        if len(mat1[0]) != len(mat2[0]):
            return None
        else:
            return mat1 + mat2
    elif axis == 1:
        if len(mat1) != len(mat2):
            return None
        else:
            result = []

            for i in range(len(mat1)):
                result.append(mat1[i] + mat2[i])

            return result
