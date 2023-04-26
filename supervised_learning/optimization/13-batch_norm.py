#!/usr/bin/env python3
""" Batch Normalization """

import numpy as np


def batch_norm(Z, gamma, beta, epsilon):
    """ Normalizes an unactivated output of a neural network using batch
        normalization.

        Args:
            Z: (tf.Tensor) with shape (m, n) that should be normalized.
            gamma: (tf.Tensor) with shape (1, n) containing the scales used for
                   batch normalization.
            beta: (tf.Tensor) with shape (1, n) containing the offsets used for
                  batch normalization.
            epsilon: (float) small number used to avoid division by zero.

        Returns:
            A tensor of the activated output for the network.
    """
    mean = np.mean(Z, axis=0)
    variance = np.var(Z, axis=0)

    # formula for batch normalization is:
    # (Z - mean) / np.sqrt(variance + epsilon)

    Z_norm = (Z - mean) / np.sqrt(variance + epsilon)

    # formula for gamma and beta is:
    # gamma * Z_norm + beta

    return gamma * Z_norm + beta
