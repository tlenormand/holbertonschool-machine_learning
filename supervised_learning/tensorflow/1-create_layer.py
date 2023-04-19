#!/usr/bin/env python3
"""
module containing function create_layer
"""
import tensorflow as tf


def create_layer(prev, n, activation):
    """
    function that create a layer in the neural network
    Args:
        prev: the tensor output of the previous layer
        n: the number of nodes in the layer to create
        activation: the activation function that the layer should use
    Return: the tensor output of the layer
    """
    layer_weights = \
        tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.Dense(
        n,
        activation=activation,
        kernel_initializer=layer_weights,
        name="layer"
    )

    return layer(prev)
