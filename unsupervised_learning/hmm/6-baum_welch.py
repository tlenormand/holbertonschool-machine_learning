#!/usr/bin/env python3
"""HMM module"""
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


def baum_welch(Observations, Transition, Emission, Initial, iterations=1000):
    """ performs the Baum-Welch algorithm for a hidden markov model

    Arguments:
        Observations: np arr (T,) index of observation(s)
        Transition: np arr (N, N) transition probabilities
        Emission: np arr (N, M) emission prob of specific observation
            given hidden state
        Initial: np arr (N, 1) prob of starting in specific hidden state
        iterations: num iter to perform

    Returns:
        Transition, Emission, or None, None on failure
    """
    if not isinstance(Observations, np.ndarray) or len(
        Observations.shape
    ) != 1:
        return None, None
    if not isinstance(Emission, np.ndarray) or len(Emission.shape) != 2:
        return None, None

    T, = Observations.shape
    N, M = Emission.shape

    if not isinstance(Transition, np.ndarray) or Transition.shape != (N, N):
        return None, None
    if np.any(np.sum(Transition, axis=1) != 1):
        return None, None
    if not isinstance(Initial, np.ndarray) or Initial.shape != (N, 1):
        return None, None
    if np.sum(Initial) != 1:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    # iterate through the number of iterations
    for i in range(iterations):
        P, F = forward(Observations, Emission, Transition, Initial)
        _, B = backward(Observations, Emission, Transition, Initial)
        gamma = (F * B) / P
        xi = np.zeros((N, N, T - 1))

        # iterate through the time steps
        for t in range(T - 1):
            obs = Observations[t + 1]

            # iterate through the states
            for i in range(N):

                # iterate through the states
                for j in range(N):
                    xi[i, j, t] = (
                        F[i, t] *
                        Transition[i, j] *
                        Emission[j, obs] *
                        B[j, t + 1]
                    ) / P

        Transition = np.sum(xi, axis=2) / np.sum(
            gamma[:, :T-1],
            axis=1
        )[..., np.newaxis]
        numerator = np.zeros((N, M))

        for k in range(M):
            numerator[:, k] = np.sum(gamma[:, Observations == k], axis=1)

        Emission = numerator / np.sum(gamma, axis=1)[..., np.newaxis]

    return Transition, Emission
