#!/usr/bin/env python3
""" Forward Propagation with Dropout """

import numpy as np


def dropout_forward_prop(X, weights, L, keep_prob):
    """ conducts forward propagation using Dropout

        Arguments:
            X: input data for the network
                nx is the number of input features
                m is the number of data points
            weights: dictionary of the weights and biases of the neural network
            L: number of layers in the network
            keep_prob: probability that a node will be kept

        Returns:
            cache: dictionary containing the outputs of each layer and the
    """
    cache = {
        'A0': X
    }

    for layer in range(L):
        A = cache['A' + str(layer)]
        W = weights['W' + str(layer + 1)]
        b = weights['b' + str(layer + 1)]
        Z = np.matmul(W, A) + b

        if layer == L - 1:
            # use softmax activation function for the last layer
            cache['A' + str(layer + 1)] = np.exp(Z) / np.sum(
                np.exp(Z), axis=0, keepdims=True
            )
        else:
            # use tanh activation function for the rest of the layers
            cache['A' + str(layer + 1)] = np.tanh(Z)

            # dropout to the hidden layer
            cache['D' + str(layer + 1)] = np.random.binomial(
                1, keep_prob, size=Z.shape
            )

            # apply dropout mask to hidden layer
            cache['A' + str(layer + 1)] *= \
                cache['D' + str(layer + 1)] / keep_prob

    return cache
