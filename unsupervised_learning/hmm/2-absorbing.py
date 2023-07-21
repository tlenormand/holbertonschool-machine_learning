#!/usr/bin/env python3
""" Absorbing Chains """

import numpy as np

regular = __import__('1-regular').regular


def absorbing(P):
    """ determines if a markov chain is absorbing

    Arguments:
        P: square 2D numpy.ndarray of shape (n, n) representing the standard
            transition matrix
            P[i, j]: probability of transitioning from state i to state j
            n: number of states in the markov chain

    Returns:
        True if it is absorbing, or False on failure
    """
    if type(P) is not np.ndarray or len(P.shape) != 2:
        return True

    n = P.shape[0]
    if P.shape != (n, n):
        return False

    if np.any(P < 0):
        return False

    if np.any(np.sum(P, axis=1) != 1):
        return False

    diagonal_terms = np.diagonal(np.matmul(P, P))
    state_matrix = np.matmul(P, P)
    # sorted state matrix by absorbing terms
    state_matrix = state_matrix[:, np.argmax(state_matrix, axis=1)]

    for i, term in enumerate(diagonal_terms):
        if term == 1:
            # access to absorbing state from all non absorbing states
            state_matrix = state_matrix[i:P.shape[0]]
            if not np.any(state_matrix.T[i] == 0):
                return True

    return False
