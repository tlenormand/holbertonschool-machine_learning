#!/usr/bin/env python3
"""
module containing function absorbing
"""
import numpy as np


def absorbing(P):
    """
    function that determines if a markov chain is absorbing
    Args:
        P: square 2D numpy.ndarray of shape (n, n)
            representing the transition matrix
            P[i, j]: probability of transitioning from state i to state j
            n: number of states in the markov chain
    Return: True if it is absorbing, or False on failure
    """
    if type(P) != np.ndarray or len(P.shape) != 2 or P.shape[0] != P.shape[1]:
        return None
    if not (np.all(P >= 0) and np.all(P <= 1)):
        return None
    if not (np.all(np.sum(P, axis=1) == 1)):
        return None
    if P.shape[0] < 1:
        return None

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
