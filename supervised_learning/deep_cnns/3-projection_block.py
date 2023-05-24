#!/usr/bin/env python3
""" Projection Block """

import tensorflow.keras as K


def projection_block(A_prev, filters, s=2):
    """ Function that builds an identity block
        as described in Deep Residual Learning for
        Image Recognition (2015)

    Arguments:
        - A_prev is the output from the previous layer
        - filters is a tuple or list containing
            F11, F3, F12, respectively:
            - F11 is the number of filters in the first 1x1 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F12 is the number of filters in the second 1x1 convolution
        - s is the stride of the first convolution in both the main path
            and the shortcut connection

    Returns:
        The activated output of the identity block
    """
    F11, F3, F12 = filters

    # =============================================================================
    # Initialize weights with he normal
    init = K.initializers.he_normal(seed=None)

    # =============================================================================
    # 1x1 Convolution
    conv2d = K.layers.Conv2D(
        filters=F11,
        strides=s,
        kernel_size=(1, 1),
        padding='same',
        activation='linear',
        kernel_initializer=init
    )(A_prev)

    batch_normalization = K.layers.BatchNormalization()(conv2d)
    activation = K.layers.Activation('relu')(batch_normalization)

    # =============================================================================
    # 1x1 Convolution
    conv2d_1 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='linear',
        kernel_initializer=init
    )(activation)

    batch_normalization_1 = K.layers.BatchNormalization()(conv2d_1)
    activation_1 = K.layers.Activation('relu')(batch_normalization_1)

    # =============================================================================
    # 1x1 Convolution
    conv2d_2 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        padding='same',
        activation='linear',
        kernel_initializer=init
    )(activation_1)

    # =============================================================================
    # 1x1 Convolution shortcut
    conv2d_3 = K.layers.Conv2D(
        filters=F12,
        kernel_size=(1, 1),
        strides=s,
        padding='same',
        activation='linear',
        kernel_initializer=init
    )(A_prev)

    batch_normalization_2 = K.layers.BatchNormalization()(conv2d_2)
    batch_normalization_3 = K.layers.BatchNormalization()(conv2d_3)

    # =============================================================================
    # Add shortcut to main path
    add = K.layers.Add()([batch_normalization_2, batch_normalization_3])

    activation_2 = K.layers.Activation('relu')(add)

    return activation_2
