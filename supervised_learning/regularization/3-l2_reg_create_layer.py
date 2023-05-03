#!/usr/bin/env python3
""" L2 Regularization create layer """

import tensorflow as tf


def l2_reg_create_layer(prev, n, activation, lambtha):
    """ Creates a tensorflow layer that includes L2 regularization

    Arguments:
        prev (tensor): tensor output of the previous layer
        n (int): number of nodes for the new layer
        activation (tensor): activation function of the layer
        lambtha (float): L2 regularization parameter

    Returns:
        tensor: output of the new layer
    """
    # mode can be FAN_IN, FAN_OUT, FAN_AVG
    # FAN_IN: the number of input units in the weight tensor
    # FAN_OUT: the number of output units in the weight tensor
    # FAN_AVG: average of FAN_IN and FAN_OUT
    layer_weights = \
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=layer_weights,
        name="layer",
        kernel_regularizer=tf.contrib.layers.l2_regularizer(lambtha)
    )

    return layer(prev)
