#!/usr/bin/env python3
""" Batch Normalization """

import tensorflow.compat.v1 as tf


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
    mean, variance = tf.nn.moments(x=Z, axes=[0], keep_dims=True)

    return tf.nn.batch_normalization(
        x=Z,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon
    )
