#!/usr/bin/env python3
""" L2 Regularization gradient descent """

import numpy as np


def l2_reg_gradient_descent(Y, weights, cache, alpha, lambtha, L):
    """ Updates the weights and biases of a neural network using gradient
        descent with L2 regularization

    Arguments:
        Y (numpy.ndarray): one-hot matrix of shape (classes, m) that contains
            the correct labels for the data
            classes: number of classes
            m: number of data points
        weights (dict): dictionary of the weights and biases of the neural
            network
        cache (dict): dictionary of the outputs of each layer of the neural
            network
        alpha (float): learning rate
        lambtha (float): L2 regularization parameter
        L (int): number of layers of the network

    Returns:
        None
    """
    m = Y.shape[1]
    # dZ = cache["A" + str(L)] - Y derivative of last layer
    dZ = cache["A" + str(L)] - Y

    for layer in range(L, 0, -1):
        A_prev = cache["A" + str(layer - 1)]

        # dW = (1 / m) * np.matmul(dZ, A_prev.T) + \
        #     ((lambtha / m) * weights["W" + str(layer)]) derivative of weights
        dW = (1 / m) * np.matmul(dZ, A_prev.T) + \
            ((lambtha / m) * weights["W" + str(layer)])

        # db = (1 / m) * np.sum(dZ, axis=1, keepdims=True) derivative of biases
        db = (1 / m) * np.sum(dZ, axis=1, keepdims=True)

        # dZ = np.matmul(weights["W" + str(layer)].T, dZ) * (1 - A_prev**2)
        # derivative of Z
        dZ = np.matmul(weights["W" + str(layer)].T, dZ) * (1 - A_prev**2)

        weights["W" + str(layer)] -= alpha * dW
        weights["b" + str(layer)] -= alpha * db