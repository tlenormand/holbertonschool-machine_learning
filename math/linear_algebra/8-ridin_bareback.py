#!/usr/bin/env python3
"""
    Functions:
        mat_mul: function that performs matrix multiplication
"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def mat_mul(mat1, mat2):
    """ function that performs matrix multiplication

    Arguments:
        mat1: matrix 1
        mat2: matrix 2

    Returns:
        new matrix multiplicated
    """
    if not mat1 or not mat2:
        return None

    shapeMat1 = matrix_shape(mat1)
    shapeMat2 = matrix_shape(mat2)

    if shapeMat1[1] != shapeMat2[0]:
        return None

    newMat = []

    for i in range(shapeMat1[0]):
        newMat.append([])
        for j in range(shapeMat2[1]):
            newMat[i].append(0)
            for k in range(shapeMat1[1]):
                newMat[i][j] += mat1[i][k] * mat2[k][j]

    return newMat
