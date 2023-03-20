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
    if (not mat1 or not mat2 or
            len(mat1) != len(mat2) or
            len(mat1[0]) != len(mat2[0])):
        return None

    return [
        [mat1[i][j] + mat2[i][j] for j in range(len(mat1))]
        for i in range(len(mat1[0]))
    ]
