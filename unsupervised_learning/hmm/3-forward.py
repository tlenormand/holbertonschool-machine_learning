#!/usr/bin/env python3
""" Forward Algorithm """

import numpy as np


def forward(Observation, Emission, Transition, Initial):
    """ performs the forward algorithm for a hidden markov model

    Arguments:
        Observation: np arr (T,) index of observation(s)
        Emission: np arr (N, M) emission prob of specific observation
            given hidden state
        Transition: np arr (N, N) transition probabilities
        Initial: np arr (N, 1) prob of starting in specific hidden state

    Returns:
        P, F, or None, None on failure
            P: likelihood of the observations given the model
            F: np arr (N, T) forward path probabilities
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

    # initialize F
    F = np.zeros((N, T))
    F[:, 0] = Initial.T * Emission.T[Observation[0]]

    for t in range(1, T):
        F[:, t] = np.matmul(F[:, t - 1], Transition) *\
            Emission.T[Observation[t]]

    P = np.sum(F[:, T - 1])

    return P, F
