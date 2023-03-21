#!/usr/bin/env python3
"""
    Functions:
        np_slice: function that slices a matrix along specific axes
"""


def np_slice(matrix, axes={}):
    """ function that slices a matrix along specific axes

    Arguments:
        matrix: numpy.ndarray
        axes: dictionary where the key is an axis to slice along and the
            value is a tuple representing the slice to make along that axis

    Returns:
        sliced matrix
    """
    slices = []

    # build array of slices
    for i in range(matrix.ndim):
        if i in axes:
            slices.append(slice(*axes[i]))
        else:
            slices.append(slice(None))

    # return sliced matrix
    return matrix[tuple(slices)]
