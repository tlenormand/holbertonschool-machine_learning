#!/usr/bin/env python3
""" Backward Algorithm """

import numpy as np


def backward(Observation, Emission, Transition, Initial):
    """ performs the backward algorithm for a hidden markov model

    Arguments:
        Observation: np arr (T,) index of observation(s)
        Emission: np arr (N, M) emission prob of specific observation
            given hidden state
        Transition: np arr (N, N) transition probabilities
        Initial: np arr (N, 1) prob of starting in specific hidden state

    Returns:
        P, B, or None, None on failure
            P: likelihood of the observations given the model
            B: np arr (N, T) containing the backward path probabilities
                B[i, j] is the probability of generating the future
                observations from hidden state i at time j
    """
    if type(Observation) is not np.ndarray or Observation.ndim != 1:
        return None, None
    if type(Emission) is not np.ndarray or Emission.ndim != 2:
        return None, None
    if type(Transition) is not np.ndarray or Transition.ndim != 2:
        return None, None
    if type(Initial) is not np.ndarray or Initial.ndim != 2:
        return None, None

    N, M = Emission.shape
    if Transition.shape != (N, N):
        return None, None
    if Initial.shape != (N, 1):
        return None, None

    T = Observation.shape[0]

    # initialization of Beta
    B = np.zeros(shape=(N, T))
    B.T[T - 1] = 1

    for t in range(T - 2, -1, -1):
        for s in range(N):
            B[s, t] = (B.T[t + 1] * Transition[s, :] *
                       Emission.T[Observation[t + 1]]).sum()

    P = np.sum(Initial.T * Emission.T[Observation[0]] * B.T[0])

    return P, B
