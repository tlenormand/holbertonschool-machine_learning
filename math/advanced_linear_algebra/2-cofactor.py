#!/usr/bin/env python3
""" Cofactor """

determinant = __import__('0-determinant').determinant
minor = __import__('1-minor').minor


def cofactor(matrix):
    """ calculates the cofactor matrix of a matrix

    Arguments:
        matrix (list): matrix to calculate

    Returns:
        the cofactor matrix of matrix
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
        return [[1]]

    minorMatrix = minor(matrix)

    sign = 1
    cofactor = []
    for i in range(len(minorMatrix)):
        if len(minorMatrix) % 2 == 0 and i % 2 != 0:
            sign *= -1

        cofactor.append([])
        for j in range(len(minorMatrix)):
            cofactor[i].append(sign * minorMatrix[i][j])
            sign *= -1

    return cofactor
