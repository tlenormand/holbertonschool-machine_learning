#!/usr/bin/env python3
""" Likelihood """

import numpy as np
from scipy import special


def likelihood(x, n, P):
    """ calculates the likelihood of obtaining this data given various
        hypothetical probabilities of developing severe side effects

    Arguments:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
            probabilities of developing severe side effects

    Returns:
        a 1D numpy.ndarray containing the likelihood of obtaining the data,
        x and n, for each probability in P, respectively
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")

    # proba = (n! / (x! * (n - x)!)) * (P ** x) * ((1 - P) ** (n - x))

    comb = np.math.factorial(n) / (
        np.math.factorial(x) * np.math.factorial(n - x)
    )

    return comb * (P ** x) * ((1 - P) ** (n - x))


def intersection(x, n, P, Pr):
    """ calculates the intersection of obtaining this data with the various
        hypothetical probabilities

    Arguments:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
            probabilities of developing severe side effects
        Pr is a 1D numpy.ndarray containing the prior beliefs of P

    Returns:
        a 1D numpy.ndarray containing the intersection of obtaining x and n
        with each probability in P, respectively
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(P) is not np.ndarray or len(P.shape) != 1:
        raise TypeError("P must be a 1D numpy.ndarray")
    if type(Pr) is not np.ndarray or len(Pr.shape) != 1 or Pr.shape != P.shape:
        raise TypeError("Pr must be a numpy.ndarray with the same shape as P")
    if np.any(P > 1) or np.any(P < 0):
        raise ValueError("All values in P must be in the range [0, 1]")
    if np.any(Pr > 1) or np.any(Pr < 0):
        raise ValueError("All values in Pr must be in the range [0, 1]")
    if P.shape != Pr.shape:
        raise TypeError("Pr must have the same shape as P")
    if not np.isclose(np.sum(Pr), 1):
        raise ValueError("Pr must sum to 1")

    return likelihood(x, n, P) * Pr


def marginal(x, n, P, Pr):
    """ calculates the marginal probability of obtaining the data

    Arguments:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
            probabilities of patients developing severe side effects
        Pr is a 1D numpy.ndarray containing the prior beliefs about P

    Returns:
        the marginal probability of obtaining x and n
    """
    return np.sum(intersection(x, n, P, Pr))


def posterior(x, n, P, Pr):
    """ calculates the posterior probability for the various hypothetical
        probabilities of developing severe side effects given the data

    Arguments:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        P is a 1D numpy.ndarray containing the various hypothetical
            probabilities of patients developing severe side effects
        Pr is a 1D numpy.ndarray containing the prior beliefs of P

    Returns:
        the posterior probability of each probability in P given x and n
    """
    return intersection(x, n, P, Pr) / marginal(x, n, P, Pr)


def posterior(x, n, p1, p2):
    """ calculates the posterior probability that the probability of
        developing severe side effects falls within a specific range
        given the data

    Arguments:
        x is the number of patients that develop severe side effects
        n is the total number of patients observed
        p1 is the lower bound on the range
        p2 is the upper bound on the range

    Returns:
        the posterior probability that p is within the range [p1, p2]
        given x and n
    """
    if type(n) is not int or n < 1:
        raise ValueError("n must be a positive integer")
    if type(x) is not int or x < 0:
        raise ValueError(
            "x must be an integer that is greater than or equal to 0"
        )
    if x > n:
        raise ValueError("x cannot be greater than n")
    if type(p1) is not float or p1 < 0 or p1 > 1:
        raise ValueError("p1 must be a float in the range [0, 1]")
    if type(p2) is not float or p2 < 0 or p2 > 1:
        raise ValueError("p2 must be a float in the range [0, 1]")
    if p2 <= p1:
        raise ValueError("p2 must be greater than p1")

    # formula using scipy.special.btdtr is:
    # btdtr(a, b, x) = Ix(a, b) = 1 - I1-x(b, a)
    # where Ix(a, b) is the regularized incomplete beta function
    # and I1-x(b, a) is the complemented incomplete beta function

    beta1 = special.btdtr(x + 1, n - x + 1, p1)
    beta2 = special.btdtr(x + 1, n - x + 1, p2)

    return beta2 - beta1
