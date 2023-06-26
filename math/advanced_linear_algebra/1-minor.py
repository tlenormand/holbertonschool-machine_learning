#!/usr/bin/env python3
""" Minor """

determinant = __import__('0-determinant').determinant


def minor(matrix):
    """ calculates the minor matrix of a matrix

    Arguments:
        matrix (list): matrix to calculate

    Returns:
        the minor matrix of matrix
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

    minor = []
    for i in range(len(matrix)):
        minor.append([])
        for j, k in enumerate(matrix[1]):
            sub_matrix = []
            for row in matrix[:i] + matrix[i + 1:]:
                new_row = []
                for k, col in enumerate(row):
                    if k != j:
                        new_row.append(col)
                sub_matrix.append(new_row)
            minor[i].append(determinant(sub_matrix))

    return minor
