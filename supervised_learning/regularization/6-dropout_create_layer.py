#!/usr/bin/env python3
""" Create a Layer with Dropout """

import numpy as np


def dropout_create_layer(prev, n, activation, keep_prob):
    """ creates a layer of a neural network using dropout

    Arguments:
        prev: is a tensor containing the output of the previous layer
        n: is the number of nodes the new layer should contain
        activation: is the activation function that should be used on the layer
        keep_prob: is the probability that a node will be kept

    Returns:
        the output of the new layer
    """
    # use He et. al initialization for weights
    weights = np.random.randn(n, prev.shape[0]) * np.sqrt(2 / prev.shape[0])
    # use zeros for biases
    biases = np.zeros((n, 1))

    # create the layer using the ReLU activation function
    layer = np.matmul(weights, prev) + biases

    # dropout to the hidden layer
    dropout = np.random.binomial(1, keep_prob, size=layer.shape)

    # apply dropout mask to hidden layer
    layer *= dropout / keep_prob


    return activation(layer)
