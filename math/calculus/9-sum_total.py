#!/usr/bin/env python3
"""
    Functions:
        summation_i_squared: function that calculates the sum of i squared
"""


def summation_i_squared(n):
    """ function that calculates the sum of i squared

    Arguments:
        n: number of terms to sum

    Returns:
        sum of i squared
    """
    if not isinstance(n, int) or n < 1:
        return None

    return (n * (n + 1) * (2 * n + 1)) // 6
