#!/usr/bin/env python3
"""
    Functions:
        np_matmul: function that function that performs matrix multiplication
"""


import numpy as np


def np_matmul(mat1, mat2):
    """ function that function that performs matrix multiplication

    Arguments:
        mat1: first matrix
        mat2: second matrix

    Returns:
        new matrix with the product of each element
    """
    return np.matmul(mat1, mat2)
