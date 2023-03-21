#!/usr/bin/env python3
"""
    Functions:
        add_matrices2D: function that adds two matrices element-wise
"""


def add_matrices2D(mat1, mat2):
    """ function that adds two matrices element-wise

    Arguments:
        mat1: first matrice
        mat2: second matrice

    Returns:
        new matrice with the sum of each element
    """
    if len(mat1) == 0 and len(mat2) == 0:
        return []

    if (len(mat1) != len(mat2) or
            len(mat1[0]) != len(mat2[0])):
        return None

    return _deepCopy([
        [mat1[i][j] + mat2[i][j] for j in range(len(mat1))]
        for i in range(len(mat1[0]))
    ])


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
