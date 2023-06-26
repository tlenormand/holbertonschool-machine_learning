#!/usr/bin/env python3
""" Inverse """

determinant = __import__('0-determinant').determinant
cofactor = __import__('2-cofactor').cofactor


def inverse(matrix):
    """ calculates the inverse of a matrix

    Arguments:
        matrix (list): matrix to calculate

    Returns:
        the inverse of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')
    for row in matrix:
        if len(row) != len(matrix):
            raise ValueError('matrix must be a non-empty square matrix')

    if len(matrix) == 1:
        return [[1 / matrix[0][0]]]

    det = determinant(matrix)
    if det == 0:
        return None

    if len(matrix) == 2:
        return [[matrix[1][1] / det, -1 * matrix[0][1] / det],
                [-1 * matrix[1][0] / det, matrix[0][0] / det]]

    cofactorMatrix = cofactor(matrix)
    inverse = []
    for i in range(len(cofactorMatrix)):
        inverse.append([])
        for j in range(len(cofactorMatrix)):
            inverse[i].append(cofactorMatrix[j][i] / det)

    return inverse
