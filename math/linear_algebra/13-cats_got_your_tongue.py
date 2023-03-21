#!/usr/bin/env python3
"""
    Functions:
        np_cat: function that function that concatenates two matrices
            along a specific axis
"""


import numpy as np


def np_cat(mat1, mat2, axis=0):
    """ function that function that concatenates two matrices
        along a specific axis

    Arguments:
        mat1: first matrix
        mat2: second matrix
        axis: axis to concatenate

    Returns:
        new matrix concatenate
    """
    return np.concatenate((mat1, mat2), axis=axis)
