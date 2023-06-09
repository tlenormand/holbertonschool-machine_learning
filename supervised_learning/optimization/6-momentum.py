#!/usr/bin/env python3
""" Momentum """

import tensorflow as tf


def create_momentum_op(loss, alpha, beta1):
    """ Creates the training operation for a neural network in tensorflow
        using the gradient descent with momentum optimization algorithm.

        Args:
            loss: (tf.Tensor) the loss of the network
            alpha: (float) the learning rate
            beta1: (float) the momentum weight

        Returns:
            The momentum optimization operation.
    """
    return tf.train.MomentumOptimizer(alpha, beta1).minimize(loss)
