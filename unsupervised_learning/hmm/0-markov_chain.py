#!/usr/bin/env python3
""" Markov Chain """

import numpy as np


def markov_chain(P, s, t=1):
    """ determines the probability of a markov chain being in a particular
        state after a specified number of iterations

    Arguments:
        P: square 2D numpy.ndarray of shape (n, n) representing the transition
            matrix
            P[i, j]: probability of transitioning from state i to state j
            n: number of states in the markov chain
        s: numpy.ndarray of shape (1, n) representing the probability of
            starting in each state
        t: number of iterations that the markov chain has been through

    Returns:
        numpy.ndarray of shape (1, n) representing the probability of
            being in a specific state after t iterations, or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None
    if type(t) is not int or t < 1:
        return None

    n = P.shape[0]
    if P.shape != (n, n):
        return None
    if type(s) is not np.ndarray or len(s.shape) != 2:
        return None
    if s.shape != (1, n):
        return None

    for _ in range(t):
        s = np.matmul(s, P)

    return s
