#!/usr/bin/env python3
"""
    Functions:
        summation_i_squared: function that calculates the integral
            of a polynomial
"""


def poly_integral(poly, C=0):
    """ function that calculates the integral of a polynomial

    Arguments:
        poly: list of coefficients representing a polynomial
        C: integration constant

    Returns:
        list of coefficients representing the integral of the polynomial
    """
    if not poly or len(poly) == 0:
        return None

    result = [C]

    for i in range(len(poly)):
        operation = poly[i] / (i + 1)
        if not(operation == 0 and len(result) == 1):
            result.append(operation)

    if len(result) == 1:
        return None

    return result
