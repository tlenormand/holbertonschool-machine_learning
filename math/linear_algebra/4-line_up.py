#!/usr/bin/env python3
"""
    Functions:
        add_arrays: function that adds two arrays element-wise
"""


def add_arrays(arr1, arr2):
    """ function that adds two arrays element-wise

    Arguments:
        arr1: first array
        arr2: second array

    Returns:
        new array with the sum of each element
    """
    if len(arr1) != len(arr2):
        return None

    return [arr1[i] + arr2[i] for i in range(len(arr1))]
