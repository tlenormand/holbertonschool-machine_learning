#!/usr/bin/env python3
""" RNN """

import numpy as np


class RNNCell:
    """ Class that represents a cell of a simple RNN """

    def __init__(self, i, h, o):
        """ Constructor

        Arguments:
            i {int} -- Is the dimensionality of the data
            h {int} -- Is the dimensionality of the hidden state
            o {int} -- Is the dimensionality of the outputs

        Returns:
            None
        """
        self.Wh = np.random.randn(h + i, h)  # concatenation of h and i
        self.Wy = np.random.randn(h, o)  # output
        self.bh = np.zeros((1, h))
        self.by = np.zeros((1, o))

    @staticmethod
    def softmax(z):
        """ Function that performs softmax over a numpy.ndarray matrix

        Arguments:
            z {numpy.ndarray} -- Is a matrix to perform softmax over
                shape (m, n) where
                    m is the number of rows
                    n the number

        Returns:
            numpy.ndarray -- Matrix with softmax performed
        """
        return np.exp(z) / np.sum(np.exp(z), axis=1, keepdims=True)

    def forward(self, h_prev, x_t):
        """ Function that performs forward propagation for one time step

        Arguments:
            h_prev {numpy.ndarray} -- Contains the previous hidden state
                shape (m, h)
                    m is the number of rows
                    h is the dimensionality of the hidden state
            x_t {numpy.ndarray} -- Contains the data input of the cell
                shape (m, i)
                    m is the number of rows
                    i is the dimensionality of the data

        Returns:
            tuple(numpy.ndarray) -- Contains the next hidden state, the output
                of the cell
        """
        # Concat h_prev and x_t to match Wh dimensions
        x_concat = np.concatenate((h_prev, x_t), axis=1)

        # Compute next hidden state
        h_next = np.tanh(np.matmul(x_concat, self.Wh) + self.bh)

        # Compute output of the cell
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, y
