#!/usr/bin/env python3
""" RNN """

import numpy as np


def rnn(rnn_cell, X, h_0):
    """ Function that performs forward propagation for a simple RNN

    Arguments:
        rnn_cell {RNNCell} -- Is the instance of RNNCell that will be used
            for the forward propagation
        X {numpy.ndarray} -- Contains the data to be used
            shape (t, m, i)
                t is the maximum number of time steps
                m is the batch size
                i is the dimensionality of the data
        h_0 {numpy.ndarray} -- Contains the initial hidden state
            shape (m, h)
                m is the batch size
                h is the dimensionality of the hidden state

    Returns:
        H, Y {tuple} -- Contains all the hidden states and outputs
            H {numpy.ndarray} -- Contains all the hidden states
            Y {numpy.ndarray} -- Contains all the outputs
    """
    # Get dimensions
    t, m, i = X.shape
    _, h = h_0.shape

    # Initialize h_prev and y
    h_prev = h_0
    H = np.zeros((t + 1, m, h))
    Y = np.zeros((t, m, rnn_cell.Wy.shape[1]))

    # Compute h_next and y for each time step
    for step in range(t):
        h_next, y = rnn_cell.forward(h_prev, X[step])
        h_prev = h_next
        H[step + 1] = h_next
        Y[step] = y

    return H, Y
