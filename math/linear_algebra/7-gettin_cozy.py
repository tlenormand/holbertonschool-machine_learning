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
    result = []

    if axis == 0:
        result = mat1 + mat2
        return _deepCopy(result)
    else:
        for i in range(len(mat1)):
            inter = cat_matrices2D(mat1[i], mat2[i], axis - 1)
            result.append(inter)
    return _deepCopy(result)


def _deepCopy(arr):
    """ function that deep copy an array

    Arguments:
        arr: array to copy

    Returns:
        new array copy
    """
    if isinstance(arr, list):
        result = []

        for i in arr:
            result.append(_deepCopy(i))
    elif isinstance(arr, (int, float, type(None), str, bool)):
        result = arr
    else:
        return None

    return result
