#!/usr/bin/env python3
""" DenseNet-121 """

import tensorflow.keras as K

dense_block = __import__('5-dense_block').dense_block
transition_layer = __import__('6-transition_layer').transition_layer


def densenet121(growth_rate=32, compression=1.0):
    """ builds the DenseNet-121 architecture as described in
            Densely Connected Convolutional Networks

    Arguments:
        growth_rate: is the growth rate
        compression: is the compression factor

    Returns:
        the keras model
    """
    initializer = K.initializers.he_normal(seed=None)
    X = K.Input(shape=(224, 224, 3))

    # =============================================================================
    # 7x7 Convolution
    batch_normalization = K.layers.BatchNormalization()(X)
    activation = K.layers.Activation('relu')(batch_normalization)
    conv2d = K.layers.Conv2D(
        filters=64,
        kernel_size=(7, 7),
        strides=(2, 2),
        padding='same',
        activation='linear',
        kernel_initializer=initializer
    )(activation)

    # =============================================================================
    # 2D Max Pooling
    max_pool = K.layers.MaxPooling2D(
        pool_size=(3, 3),
        strides=(2, 2),
        padding='same'
    )(conv2d)

    # =============================================================================
    # Dense Block 1
    dense1, nb_filters = dense_block(max_pool, 64, growth_rate, 6)

    # =============================================================================
    # Transition Layer 1
    transition1, nb_filters = transition_layer(dense1, nb_filters, compression)

    # =============================================================================
    # Dense Block 2
    dense2, nb_filters = dense_block(transition1, nb_filters, growth_rate, 12)

    # =============================================================================
    # Transition Layer 2
    transition2, nb_filters = transition_layer(dense2, nb_filters, compression)

    # =============================================================================
    # Dense Block 3
    dense3, nb_filters = dense_block(transition2, nb_filters, growth_rate, 24)

    # =============================================================================
    # Transition Layer 3
    transition3, nb_filters = transition_layer(dense3, nb_filters, compression)

    # =============================================================================
    # Dense Block 4
    dense4, nb_filters = dense_block(transition3, nb_filters, growth_rate, 16)

    # =============================================================================
    # 2D Average Pooling
    avg_pool = K.layers.AveragePooling2D(
        pool_size=(7, 7),
        strides=(1, 1),
        padding='valid'
    )(dense4)

    # =============================================================================
    # Fully Connected Softmax Output
    FC_softmax = K.layers.Dense(
        units=1000,
        activation='softmax',
        kernel_initializer=initializer
    )(avg_pool)

    # =============================================================================
    # Create Keras Model instance
    model = K.models.Model(inputs=X, outputs=FC_softmax)

    return model
