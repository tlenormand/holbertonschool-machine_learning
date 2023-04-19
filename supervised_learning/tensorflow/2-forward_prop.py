#!/usr/bin/env python3
"""
module containing function forward_prop
"""
import tensorflow as tf

create_layer = __import__('1-create_layer').create_layer


def forward_prop(x, layer_sizes=[], activations=[]):
    """
    function that create creates the forward propagation graph
        for the neural network
    Args:
        x: the placeholder for the input data
        layer_sizes: list containing the number of nodes in each layer
            of the network
        activations: list containing the activation functions
            for each layer of the network
    Return: the prediction of the network in tensor form
    """
    for i in range(len(layer_sizes)):
        x = create_layer(x, layer_sizes[i], activations[i])

    return x
