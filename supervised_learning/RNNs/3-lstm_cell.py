#!/usr/bin/env python3
""" RNN """

import numpy as np


class LSTMCell:
    """ Long Short Term Memory """

    def __init__(self, i, h, o):
        """ Constructor

        Arguments:
            i {int} -- Is the dimensionality of the data
            h {int} -- Is the dimensionality of the hidden state
            o {int} -- Is the dimensionality of the outputs

        Returns:
            None
        """
        self.Wf = np.random.normal(size=(i + h, h))
        self.Wu = np.random.normal(size=(i + h, h))
        self.Wc = np.random.normal(size=(i + h, h))
        self.Wo = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h, o))
        self.bf = np.zeros((1, h))
        self.bu = np.zeros((1, h))
        self.bc = np.zeros((1, h))
        self.bo = np.zeros((1, h))
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

    def forward(self, h_prev, c_prev, x_t):
        """ Forward Propagation

        Arguments:
            h_prev {np.ndarray} -- Contains the previous hidden state
            c_prev {np.ndarray} -- Contains the previous cell state
            x_t {np.ndarray} -- Contains the data input of the cell

        Returns:
            tuple(np.ndarray) -- Contains the next hidden state, the next
                cell state and the output
                h_next {np.ndarray} -- Next hidden state
                c_next {np.ndarray} -- Next cell state
                y {np.ndarray} -- Output of the cell
        """
        # concatenate previous hidden state and input
        x = np.concatenate((h_prev, x_t), axis=1)

        # forget gate
        f = self.sigmoid(np.matmul(x, self.Wf) + self.bf)

        # update gate
        u = self.sigmoid(np.matmul(x, self.Wu) + self.bu)

        # calculate the new information of the cell state
        c = np.tanh(np.matmul(x, self.Wc) + self.bc)

        # new cell state
        c_next = f * c_prev + u * c

        # output gate
        o = self.sigmoid(np.matmul(x, self.Wo) + self.bo)

        # new hidden state
        h_next = o * np.tanh(c_next)

        # output of the cell
        y = self.softmax(np.matmul(h_next, self.Wy) + self.by)

        return h_next, c_next, y
