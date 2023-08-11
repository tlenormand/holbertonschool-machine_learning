#!/usr/bin/env python3
""" RNN """

import numpy as np


def deep_rnn(rnn_cells, X, h_0):
    """ Performs forward propagation for a deep RNN

    Arguments:
        rnn_cells {list} -- Is the list of RNNCell instances
        X {np.ndarray} -- Contains the data to be used
        h_0 {np.ndarray} -- Contains the initial hidden state

    Returns:
        tuple -- Contains all the hidden states and the outputs
    """
    t, m, _ = X.shape
    l, _, h = h_0.shape
    Y = []

    H = np.zeros((t + 1, l, m, h))
    H[0] = h_0

    for step in range(t):
        for layer in range(l):
            if layer == 0:
                h, y = rnn_cells[layer].forward(
                    H[step][layer],
                    X[step]
                )
            else:
                h, y = rnn_cells[layer].forward(
                    H[step][layer],
                    h
                )

            H[step + 1][layer] = h
        Y.append(y)

    return H, np.array(Y)
