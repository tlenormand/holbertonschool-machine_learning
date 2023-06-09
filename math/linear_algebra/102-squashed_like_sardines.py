#!/usr/bin/env python3
"""
    Functions:
        cat_matrices: function that concatenates two matrices
            along a specific axis
"""


matrix_shape = __import__('2-size_me_please').matrix_shape


def cat_matrices(mat1, mat2, axis=0):
    """ function that concatenates two matrices along a specific axis

    Arguments:
        mat1: first matrix
        mat2: second matrix
        axis: axis to concatenate

    Returns:
        new matrix with the concatenation of each element
        None if the matrices cannot be concatenated
    """
    shape_mat1 = matrix_shape(mat1)
    shape_mat2 = matrix_shape(mat2)

    # check dimention are identical or axis not too deep
    if (len(shape_mat1) != len(shape_mat2) or
            axis >= len(shape_mat1)):
        return None

    # check shape of the matrices are identical
    for i in range(len(shape_mat1)):
        if axis != i and shape_mat1[i] != shape_mat2[i]:
            return None

    if axis == 0:
        # if we are on the first axis, we concatenate the matrices
        return mat1 + mat2
    else:
        # if we are not on the first axis, recursively concatenate the matrices
        result = [
            cat_matrices(mat1[i], mat2[i], axis - 1) for i in range(len(mat1))
        ]

        return (None if None in result else result)
