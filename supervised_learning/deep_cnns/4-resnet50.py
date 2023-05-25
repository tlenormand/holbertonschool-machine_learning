#!/usr/bin/env python3
""" ResNet-50 """

import tensorflow.keras as K

identity_block = __import__('2-identity_block').identity_block
projection_block = __import__('3-projection_block').projection_block


def resnet50():
    """ builds the ResNet-50 architecture as described in
            Deep Residual Learning for Image Recognition (2015)

    Arguments:
        None

    Returns:
        the keras model
    """
    input_layer = K.layers.Input(shape=(224, 224, 3))

    # =============================================================================
    # Convolutional layer with 7x7 kernel and stride of 2x2
    conv1 = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        activation='linear'
    )(input_layer)

    batch_normalization = K.layers.BatchNormalization()(conv1)
    activation = K.layers.ReLU()(batch_normalization)

    # =============================================================================
    # Max pooling layer with kernels of shape 3x3 and default strides
    max_pool1 = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(activation)

    # =============================================================================
    # Projection block
    projection = projection_block(max_pool1, [64, 64, 256], 1)
    identity = identity_block(projection, [64, 64, 256])
    identity = identity_block(identity, [64, 64, 256])

    # =============================================================================
    # Projection block
    projection = projection_block(identity, [128, 128, 512])
    identity = identity_block(projection, [128, 128, 512])
    identity = identity_block(identity, [128, 128, 512])
    identity = identity_block(identity, [128, 128, 512])

    # =============================================================================
    # Projection block
    projection = projection_block(identity, [256, 256, 1024])
    identity = identity_block(projection, [256, 256, 1024])
    identity = identity_block(identity, [256, 256, 1024])
    identity = identity_block(identity, [256, 256, 1024])
    identity = identity_block(identity, [256, 256, 1024])
    identity = identity_block(identity, [256, 256, 1024])

    # =============================================================================
    # Projection block
    projection = projection_block(identity, [512, 512, 2048])
    identity = identity_block(projection, [512, 512, 2048])
    identity = identity_block(identity, [512, 512, 2048])

    # =============================================================================
    # Average pooling layer with kernels of shape 7x7
    avg_pool = K.layers.AveragePooling2D(
        strides=(7, 7),
        padding='same'
    )(identity)

    # =============================================================================
    # Fully connected layer with softmax activation
    softmax = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=K.initializers.he_normal(seed=None)
    )(avg_pool)

    # =============================================================================
    # Create model
    model = K.models.Model(inputs=input_layer, outputs=softmax)

    return model
