#!/usr/bin/env python3
"""
    Functions:
        summation_i_squared: function that calculates the derivative
            of a polynomial
"""


def poly_derivative(poly):
    """ function that calculates the derivative of a polynomial

    Arguments:
        poly: list of coefficients representing a polynomial

    Returns:
        list of coefficients representing the derivative of the polynomial
    """
    if not poly or not isinstance(poly, list) or len(poly) == 0:
        return None

    result = []

    for i in range(len(poly)):
        operation = poly[i] * i
        if (operation != 0 or len(poly) != i):
            result.append(operation)

    if len(result) == 0:
        return [0]

    return result
