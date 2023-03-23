#!/usr/bin/env python3
"""
    Functions:
        cat_matrices: function that concatenates two matrices along a specific axis
"""


matrix_shape = __import__('2-size_me_please').matrix_shape
cat_matrices2D = __import__('7-gettin_cozy').cat_matrices2D


def cat_matrices(mat1, mat2, axis=0):
    """ function that concatenates two matrices along a specific axis

    Arguments:
        mat1: first matrix
        mat2: second matrix
        axis: axis to concatenate

    Returns:
        new matrix with the concatenation of each element
    """
    shape_mat1 = matrix_shape(mat1)
    shape_mat2 = matrix_shape(mat2)

    if len(shape_mat1) != len(shape_mat2):
        return None

    if len(shape_mat1) == len(shape_mat2) == 1:
        return mat1 + mat2
    
    if axis > 1:
        return [
            cat_matrices(mat1[i], mat2[i], axis - 1) for i in range(len(mat1))
        ]
    else:
        return cat_matrices2D(mat1, mat2, axis)
