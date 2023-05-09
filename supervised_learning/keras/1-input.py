#!/usr/bin/env python3
""" Input """

import tensorflow.keras as K


def build_model(nx, layers, activations, lambtha, keep_prob):
    """ Function that builds a neural network with the Keras library

    Arguments:
        nx: is the number of input features to the network
        layers: is a list containing the number of nodes in each layer of the
            network
        activations: is a list containing the activation functions used for
            each layer of the network
        lambtha: is the L2 regularization parameter
        keep_prob: is the probability that a node will be kept for dropout

    Returns:
        the keras model
    """
    # initialize model with input class
    x = K.Input(shape = (nx,))

    # first layer
    y = K.layers.Dense(
        layers[0],
        activation = activations[0],
        kernel_regularizer = K.regularizers.l2(lambtha)
    )(x)

    y = K.layers.Dropout(1 - keep_prob)(y)

    # next layers
    for layer in range(len(layers)):
        y = K.layers.Dense(
            layers[layer],
            activation = activations[layer],
            kernel_regularizer = K.regularizers.l2(lambtha)
        )(y)

        if layer < len(layers) - 1:
            y = K.layers.Dropout(1 - keep_prob)(y)

    # create model
    model = K.Model(inputs = x, outputs = y)

    return model