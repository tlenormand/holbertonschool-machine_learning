#!/usr/bin/env python3
""" Transition Layer """

import tensorflow.keras as K


def transition_layer(X, nb_filters, compression):
    """ builds a transition layer as described
            in Densely Connected Convolutional Networks

    Arguments:
        X: is the output from the previous layer
        nb_filters: is an integer representing the number of filters
                    in X
        compression: is the compression factor for the transition layer

    Returns:
        The output of the transition layer and the number of filters
    """
    init = K.initializers.he_normal(seed=None)
    filters = int(nb_filters * compression)

    batch_normalization = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_normalization)

    # =============================================================================
    # 1x1 Convolution
    conv = K.layers.Conv2D(
        filters, kernel_size=(1, 1),
        padding='same',
        kernel_initializer=init
    )(activation)

    # =============================================================================
    # 2D Average Pooling
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
        padding='valid'
    )(conv)

    return avg_pool, filters
