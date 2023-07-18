#!/usr/bin/env python3
""" Regular Chains """

import numpy as np


def regular(P):
    """ determines the steady state probabilities of a regular markov chain

    Arguments:
        P: square 2D numpy.ndarray of shape (n, n) representing the transition
            matrix
            P[i, j]: probability of transitioning from state i to state j
            n: number of states in the markov chain

    Returns:
        numpy.ndarray of shape (1, n) containing the steady state probabilities
            or None on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return None

    n = P.shape[0]
    if P.shape != (n, n):
        return None

    if np.any(P <= 0):
        return None
    if np.any(np.sum(P, axis=1) != 1):
        return None

    prob = np.ones((1, n)) / n

    # until no improvement
    while True:
        prob_prev = prob
        prob = np.matmul(prob, P)

        if np.all(prob == prob_prev):
            return prob
