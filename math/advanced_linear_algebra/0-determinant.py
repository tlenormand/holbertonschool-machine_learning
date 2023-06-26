#!/usr/bin/env python3
""" Determinant """


def determinant(matrix):
    """ calculates the determinant of a matrix

    Arguments:
        matrix (list): matrix to calculate

    Returns:
        the determinant of matrix
    """
    if type(matrix) is not list or len(matrix) == 0:
        raise TypeError('matrix must be a list of lists')
    for row in matrix:
        if type(row) is not list:
            raise TypeError('matrix must be a list of lists')

    if matrix == [[]]:
        return 1

    if len(matrix) == 1:
        return matrix[0][0]

    if len(matrix) == 2:
        return matrix[0][0] * matrix[1][1] - matrix[1][0] * matrix[0][1]

    det = 0
    sign = 1
    for i, k in enumerate(matrix[0]):
        sub_matrix = []
        for row in matrix[1:]:
            new_row = []
            for j, col in enumerate(row):
                if j != i:
                    new_row.append(col)
            sub_matrix.append(new_row)
        det += sign * k * determinant(sub_matrix)
        sign *= -1

    return det
