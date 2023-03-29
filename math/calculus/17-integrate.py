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
    if (not poly or
            not isinstance(poly, list) or
            len(poly) == 0 or
            not isinstance(C, int)):
        return None

    result = [C]

    if len(poly) == 1:
        return result

    for i in range(len(poly)):
        operation = poly[i] / (i + 1)
        result.append(int(operation) if operation.is_integer() else operation)

    return result
