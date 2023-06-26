#!/usr/bin/env python3
""" Minor """

determinant = __import__('0-determinant').determinant
minor = __import__('1-minor').minor


def adjugate(matrix):
    """ calculates the adjugate matrix of a matrix

    Arguments:
        matrix (list): matrix to calculate

    Returns:
        the adjugate matrix of matrix
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
    adjugate = []
    for i in range(len(minorMatrix)):
        adjugate.append([])
        for j in range(len(minorMatrix)):
            adjugate[i].append(sign * minorMatrix[j][i])
            sign *= -1

    return adjugate
