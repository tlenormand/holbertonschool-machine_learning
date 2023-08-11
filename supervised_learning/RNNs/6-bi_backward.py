#!/usr/bin/env python3
""" RNN """

import numpy as np


class BidirectionalCell:
    """ represents a bidirectional cell of an RNN """

    def __init__(self, i, h, o):
        """ Class constructor
            i is the dimensionality of the data
            h is the dimensionality of the hidden states
            o is the dimensionality of the outputs
            Creates the public instance attributes Whf, Whb, Wy, bhf, bhb, by
                that represent the weights and biases of the cell
                Whf and bhfare for the hidden states in the forward direction
                Whb and bhbare for the hidden states in the backward direction
                Wy and byare for the outputs
            The weights should be initialized using a random normal
                distribution in the order listed above
            The weights will be used on the right side for matrix
                multiplication
            The biases should be initialized as zeros
        """
        # weights
        self.Whf = np.random.normal(size=(i + h, h))
        self.Whb = np.random.normal(size=(i + h, h))
        self.Wy = np.random.normal(size=(h * 2, o))

        # biases
        self.bhf = np.zeros((1, h))
        self.bhb = np.zeros((1, h))
        self.by = np.zeros((1, o))

    def forward(self, h_prev, x_t):
        """ calculates the hidden state in the forward direction for one time
                step
            h_prev is a numpy.ndarray of shape (m, h) containing the previous
                hidden state
            x_t is a numpy.ndarray of shape (m, i) that contains the data input
                for the cell
                m is the batch size for the data
            The output of the cell should use a softmax activation function
            Returns: h_next, the next hidden state
        """
        # concatenate the previous hidden state and the input
        x = np.concatenate((h_prev, x_t), axis=1)

        # compute the next hidden state
        h_next = np.tanh(np.matmul(x, self.Whf) + self.bhf)

        return h_next

    def backward(self, h_next, x_t):
        """ calculates the hidden state in the backward direction for one time
                step
            h_next is a numpy.ndarray of shape (m, h) containing the next
                hidden state
                m is the batch size for the data
            x_t is a numpy.ndarray of shape (m, i) that contains the data input
                for the cell
            Returns: h_pev, the previous hidden state
        """
        # concatenate the next hidden state and the input
        x = np.concatenate((h_next, x_t), axis=1)

        # compute the previous hidden state
        h_pev = np.tanh(np.matmul(x, self.Whb) + self.bhb)

        return h_pev
