#!/usr/bin/env python3
""" Inception Network """

import tensorflow.keras as K
inception_block = __import__('0-inception_block').inception_block


def inception_network():
    """ Function that builds an inception block
        as described in Going Deeper with
        Convolutions (2014)

    Returns:
        The keras model
    """
    # =============================================================================
    # Input layer
    input_layer = K.layers.Input(shape=(224, 224, 3))

    # =============================================================================
    # Convolutional layer with 7x7 kernel and stride of 2x2
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        activation='relu'
    )(input_layer)

    # =============================================================================
    # Max pooling layer with kernels of shape 3x3 and default strides
    max_pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(conv1)

    # =============================================================================
    # Convolutional layer with 3x3 kernel and stride of 1x1
    conv2 = K.layers.Conv2D(
        filters=192,
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(max_pool1)

    # =============================================================================
    # Max pooling layer with kernels of shape 3x3 and default strides
    max_pool2 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(conv2)

    # =============================================================================
    # Inception block 3a
    inception3a = inception_block(
        max_pool2,
        [64, 96, 128, 16, 32, 32]
    )

    # =============================================================================
    # Inception block 3b
    inception3b = inception_block(
        inception3a,
        [128, 128, 192, 32, 96, 64]
    )

    # =============================================================================
    # Max pooling layer with kernels of shape 3x3 and default strides
    max_pool3 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(inception3b)

    # =============================================================================
    # Inception block 4a
    inception4a = inception_block(
        max_pool3,
        [192, 96, 208, 16, 48, 64]
    )

    # =============================================================================
    # Inception block 4b
    inception4b = inception_block(
        inception4a,
        [160, 112, 224, 24, 64, 64]
    )

    # =============================================================================
    # Inception block 4c
    inception4c = inception_block(
        inception4b,
        [128, 128, 256, 24, 64, 64]
    )

    # =============================================================================
    # Inception block 4d
    inception4d = inception_block(
        inception4c,
        [112, 144, 288, 32, 64, 64]
    )

    # =============================================================================
    # Inception block 4e
    inception4e = inception_block(
        inception4d,
        [256, 160, 320, 32, 128, 128]
    )

    # =============================================================================
    # Max pooling layer with kernels of shape 3x3 and default strides
    max_pool4 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(inception4e)

    # =============================================================================
    # Inception block 5a
    inception5a = inception_block(
        max_pool4,
        [256, 160, 320, 32, 128, 128]
    )

    # =============================================================================
    # Inception block 5b
    inception5b = inception_block(
        inception5a,
        [384, 192, 384, 48, 128, 128]
    )

    # =============================================================================
    # Average pooling layer with kernels of shape 7x7 and default strides
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        padding='valid'
    )(inception5b)

    # =============================================================================
    # Dropout (40%)
    dropout = K.layers.Dropout(
        rate=0.4
    )(avg_pool)

    # =============================================================================
    # Linear layer with 1000 units
    linear = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=K.initializers.he_normal()
    )(dropout)

    # =============================================================================
    # # Softmax activation
    # softmax = K.layers.Activation(
    #     activation='softmax'
    # )(linear)

    # =============================================================================
    # Model creation
    model = K.models.Model(inputs=input_layer, outputs=linear)

    return model
