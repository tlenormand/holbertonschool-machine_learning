#!/usr/bin/env python3
"""
    Functions:
        cat_matrices2D: function that concatenates two matrices
            along a specific axis
        _deepCopy: function that deep copy an array
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
    if not mat1 or not mat2:
        return None

    result = []

    if axis == 0:
        result.append(mat1 + mat2)
        return result
    else:
        for i in range(len(mat1)):
            inter = cat_matrices2D(mat1[i], mat2[i], axis - 1)
            result.append(inter)
    return result
