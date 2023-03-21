#!/usr/bin/env python3
"""
    Functions:
        np_elementwise: function that performs element-wise addition,
            subtraction, multiplication, and division
"""


def np_elementwise(mat1, mat2):
    """ function that function that performs element-wise addition,
        subtraction, multiplication, and division

    Arguments:
        mat1: first matrix
        mat2: second matrix

    Returns:
        tuple with the element-wise sum, difference, product, and quotient,
            respectively
    """
    return (add(mat1, mat2), sub(mat1, mat2), mul(mat1, mat2), div(mat1, mat2))


def add(mat1, mat2):
    """ function that function that performs element-wise addition

    Arguments:
        mat1: first matrix
        mat2: second matrix

    Returns:
        new matrix with the sum of each element
    """
    return mat1 + mat2


def sub(mat1, mat2):
    """ function that function that performs element-wise subtraction

    Arguments:
        mat1: first matrix
        mat2: second matrix

    Returns:
        new matrix with the difference of each element
    """
    return mat1 - mat2


def mul(mat1, mat2):
    """ function that function that performs element-wise multiplication

    Arguments:
        mat1: first matrix
        mat2: second matrix

    Returns:
        new matrix with the product of each element
    """
    return mat1 * mat2


def div(mat1, mat2):
    """ function that function that performs element-wise division

    Arguments:
        mat1: first matrix
        mat2: second matrix

    Returns:
        new matrix with the quotient of each element
    """
    return mat1 / mat2
