#!/usr/bin/env python3
""" Dense Block """

import tensorflow.keras as K


def dense_block(X, nb_filters, growth_rate, layers):
    """ builds a dense block as described in
            Densely Connected Convolutional Networks

    Arguments:
        X: is the output from the previous layer
        nb_filters: is an integer representing the number of filters
                    in X
        growth_rate: is the growth rate for the dense block
        layers: is the number of layers in the dense block

    Returns:
        The concatenated output of each layer within the Dense Block
        and the number of filters within the concatenated outputs,
        respectively
    """
    # =============================================================================
    # Initialize weights with he normal
    init = K.initializers.he_normal(seed=None)

    # =============================================================================
    # Dense Block
    for layer in range(layers):
        # =============================================================================
        # Bottleneck layer
        batch_normalization = K.layers.BatchNormalization()(X)
        activation = K.layers.Activation('relu')(batch_normalization)
        conv2d = K.layers.Conv2D(
            filters=4 * growth_rate,
            kernel_size=(1, 1),
            padding='same',
            activation='linear',
            kernel_initializer=init
        )(activation)

        # =============================================================================
        # 3x3 Convolution
        batch_normalization_1 = K.layers.BatchNormalization()(conv2d)
        activation_1 = K.layers.Activation('relu')(batch_normalization_1)
        conv2d_1 = K.layers.Conv2D(
            filters=growth_rate,
            kernel_size=(3, 3),
            padding='same',
            activation='linear',
            kernel_initializer=init
        )(activation_1)

        # =============================================================================
        # Concatenated output
        X = K.layers.concatenate([X, conv2d_1])
        nb_filters += growth_rate

    return X, nb_filters
