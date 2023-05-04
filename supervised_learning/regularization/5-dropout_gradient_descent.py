#!/usr/bin/env python3
""" Gradient Descent with Dropout """

import numpy as np


def dropout_gradient_descent(Y, weights, cache, alpha, keep_prob, L):
    """ updates the weights of a neural network with Dropout regularization
        using gradient descent

    Arguments:
        Y: one-hot numpy.ndarray of shape (classes, m) that contains the
            correct labels for the data
            classes: number of classes
            m: number of data points
        weights: dictionary of the weights and biases of the neural network
        cache: dictionary of the outputs and dropout masks of each layer of
            the neural network
        alpha: learning rate
        keep_prob: probability that a node will be kept
        L: number of layers of the network

    Returns:
        None
    """
    # last layer
    dZ = cache['A' + str(L)] - Y
    m = (1 / Y.shape[1])

    for layer in range(L, 0, -1):
        # dW represents the gradient of the cost function with respect to W
        dW = (m * np.matmul(dZ, cache['A' + str(layer - 1)].T))
        # db represents the gradient of the cost function with respect to b
        db = m * np.sum(dZ, axis=1, keepdims=True)

        # dZ represents the gradient of the cost function with respect to z
        dZ = np.matmul(weights['W' + str(layer)].T, dZ)

        A = cache['A' + str(layer - 1)]

        if layer > 1:
            dZ *= (1 - np.power(A, 2)) * \
                (cache['D' + str(layer - 1)] / keep_prob)

        weights['W' + str(layer)] -= (alpha * dW)
        weights['b' + str(layer)] -= (alpha * db)
