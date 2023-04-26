#!/usr/bin/env python3
""" Batch Normalization """

import tensorflow as tf


def create_batch_norm_layer(prev, n, activation):
    """ Creates a batch normalization layer for a neural network in tensorflow.

        Args:
            prev: (tf.Tensor) containing the output of the previous layer.
            n: (int) containing the number of nodes in the layer to be created.
            activation: (tf.Operation) containing the activation function that
                        should be used on the output of the layer.

        Returns:
            A tensor of the activated output for the layer.
    """
    initializer = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.Dense(
        units=n,
        kernel_initializer=initializer,
        activation=None
    )

    Z = layer(prev)

    mean, variance = tf.nn.moments(Z, axes=[0])
    gamma = tf.Variable(tf.ones([n]), name='gamma')
    beta = tf.Variable(tf.zeros([n]), name='beta')

    Z_norm = tf.nn.batch_normalization(
        x=Z,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=1e-8
    )

    return activation(Z_norm)
