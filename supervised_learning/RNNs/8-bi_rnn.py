#!/usr/bin/env python3
""" RNN """

import numpy as np


def bi_rnn(bi_cell, X, h_0, h_t):
    """ RNN """
    t, m, i = X.shape
    _, h = h_0.shape

    Hf = np.zeros((t + 1, m, h))
    Hb = np.zeros((t + 1, m, h))

    Hf[0] = h_0
    Hb[t] = h_t

    for step in range(t):
        # Forward
        Hf[step + 1] = bi_cell.forward(Hf[step], X[step])

    for step in range(t - 1, -1, -1):
        # Backward
        Hb[step] = bi_cell.backward(Hb[step + 1], X[step])

    H = np.concatenate((Hf[1:], Hb[0:t]), axis=-1)
    Y = bi_cell.output(H)

    return H, Y
