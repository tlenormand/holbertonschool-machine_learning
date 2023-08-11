#!/usr/bin/env python3
""" RNN """

import numpy as np


class GRUCell:
    """ Gated Recurrent Unit """

    def __init__(self, i, h, o):
        """ Constructor

        Arguments:
            i {int} -- Is the dimensionality of the data
            h {int} -- Is the dimensionality of the hidden state
            o {int} -- Is the dimensionality of the outputs

        Returns:
            None
        """
        self.Wz = np.random.normal(size=(i + h, h))
        self.Wr = np.random.normal(size=(i + h, h))
        self.Wh = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bz = np.zeros((1, h))
        self.br = np.zeros((1, h))
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def sigmoid(x):
        """ Sigmoid

        Arguments:
            x {np.ndarray} -- Contains the input value

        Returns:
            {np.ndarray} -- Contains the sigmoid output
        """
        return 1 / (1 + np.exp(-x))

    @staticmethod
    def softmax(x):
        """ Softmax

        Arguments:
            x {np.ndarray} -- Contains the input value

        Returns:
            {np.ndarray} -- Contains the softmax output
        """
        return np.exp(x) / np.sum(np.exp(x), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ Forward Propagation

        Arguments:
            h_prev {np.ndarray} -- Contains the previous hidden state
            x_t {np.ndarray} -- Contains the data input of the cell

        Returns:
            tuple(np.ndarray) -- Contains the next hidden state, the output
                h_next {np.ndarray} -- Next hidden state
                y {np.ndarray} -- Output of the cell
        """
        # concatenate the previous hidden state and the input
        x = np.concatenate((h_prev, x_t), axis=1)

        # calculate the update gate
        z = self.sigmoid(np.matmul(x, self.Wz) + self.bz)

        # calculate the reset gate
        r = self.sigmoid(np.matmul(x, self.Wr) + self.br)

        # concatenate the previous hidden state and the reset gate
        x = np.concatenate((r * h_prev, x_t), axis=1)

        # calculate the candidate hidden state
        h = np.tanh(np.matmul(x, self.Wh) + self.bh)

        # calculate the next hidden state
        h_next = z * h + (1 - z) * h_prev

        # calculate the output of the cell
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
