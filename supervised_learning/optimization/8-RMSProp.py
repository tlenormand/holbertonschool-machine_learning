#!/usr/bin/env python3
""" RMSProp """

import tensorflow as tf


def create_RMSProp_op(loss, alpha, beta2, epsilon):
    """ Creates the training operation for a neural network in tensorflow
        using the RMSProp optimization algorithm.

        Args:
            loss: (tf.Tensor) the loss of the network
            alpha: (float) the learning rate
            beta2: (float) the RMSProp weight
            epsilon: (float) a small number to avoid division by zero

        Returns:
            The RMSProp optimization operation.
    """
    return tf.train.RMSPropOptimizer(
        learning_rate=alpha,
        decay=beta2,
        epsilon=epsilon
    ).minimize(loss)
