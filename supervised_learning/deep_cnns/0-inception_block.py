#!/usr/bin/env python3
""" Inception Block """

import tensorflow.keras as K


def inception_block(A_prev, filters):
    """ Function that builds an inception block
        as described in Going Deeper with
        Convolutions (2014)

    Arguments:
        - A_prev is the output from the previous layer
        - filters is a tuple or list containing
            F1, F3R, F3, F5R, F5, FPP, respectively:
            - F1 is the number of filters in the 1x1 convolution
            - F3R is the number of filters in the 1x1 convolution
                before the 3x3 convolution
            - F3 is the number of filters in the 3x3 convolution
            - F5R is the number of filters in the 1x1 convolution
                before the 5x5 convolution
            - F5 is the number of filters in the 5x5 convolution
            - FPP is the number of filters in the 1x1 convolution
                after the max pooling

    Returns:
        The concatenated output of the inception block
    """
    F1, F3R, F3, F5R, F5, FPP = filters

    # =============================================================================
    # 1x1 Convolution
    conv1 = K.layers.Conv2D(
        filters=F1,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    # =============================================================================
    # 1x1 Convolution followed by 3x3 Convolution
    conv3R = K.layers.Conv2D(
        filters=F3R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv3 = K.layers.Conv2D(
        filters=F3,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(conv3R)

    # =============================================================================
    # 1x1 Convolution followed by 5x5 Convolution
    conv5R = K.layers.Conv2D(
        filters=F5R,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(A_prev)

    conv5 = K.layers.Conv2D(
        filters=F5,
        kernel_size=(5, 5),
        padding='same',
        activation='relu'
    )(conv5R)

    # =============================================================================
    # 3x3 Max pooling followed by 1x1 Convolution
    poolP = K.layers.MaxPool2D(
        pool_size=(3, 3),
        strides=(1, 1),
        padding='same'
    )(A_prev)

    convPP = K.layers.Conv2D(
        filters=FPP,
        kernel_size=(1, 1),
        padding='same',
        activation='relu'
    )(poolP)

    # =============================================================================
    # Concatenate the outputs of the previous layers along the channel dimension
    return K.layers.concatenate([conv1, conv3, conv5, convPP])
