#!/usr/bin/env python3
""" Sequential """

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
    model = K.Sequential()

    # first layer
    model.add(K.layers.Dense(
        layers[0],
        activation=activations[0],
        kernel_regularizer=K.regularizers.l2(lambtha),
        input_shape=(nx,)
    ))

    if len(layers) > 1:
        model.add(K.layers.Dropout(1 - keep_prob))

    # next layers
    for layer in range(1, len(layers)):
        model.add(K.layers.Dense(
            layers[layer],
            activation=activations[layer],
            kernel_regularizer=K.regularizers.l2(lambtha)
        ))

        if layer < len(layers) - 1:
            model.add(K.layers.Dropout(1 - keep_prob))

    return model
