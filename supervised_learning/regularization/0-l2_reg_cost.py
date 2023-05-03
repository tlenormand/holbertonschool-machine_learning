#!/usr/bin/env python3
""" L2 Regularization Cost """

import numpy as np


def l2_reg_cost(cost, lambtha, weights, L, m):
    """ Calculates the cost of a neural network with L2 regularization

    Arguments:
        cost (numpy.ndarray): cost without L2 regularization
        lambtha (float): regularization parameter
        weights (dict): weights and biases of the neural network
        L (int): number of layers in the neural network
        m (int): number of data points used

    Returns:
        numpy.ndarray: cost of the network accounting for L2 regularization
    """
    L2_cost = 0

    for layer in range(1, L+1):
        current_weight = weights["W" + str(layer)]
        L2_cost += np.sum(np.square(current_weight))

    L2_regularization = (lambtha / (2 * m)) * L2_cost
    regularized_cost = cost + L2_regularization

    return regularized_cost
