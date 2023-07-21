#!/usr/bin/env python3
"""HMM module"""
import numpy as np

forward = __import__('3-forward').forward
backward = __import__('5-backward').backward


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
