#!/usr/bin/env python3
""" Convolutional Autoencoder """

import tensorflow.keras as keras


def autoencoder(input_dims, filters, latent_dims):
    """ Function that creates a convolutional autoencoder

    Arguments:
        input_dims {tuple} -- Is the input size
        filters {list} -- Is a list containing the number of filters for each
            convolutional layer in the encoder, respectively
        latent_dims {tuple} -- Is the latent space dimensionality

    Returns:
        tuple -- Containing the encoder, decoder and autoencoder models
    """
    ##############################
    # ENCODER MODEL
    ##############################
    encoded_inputs = keras.Input(shape=(input_dims))
    encoded = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(encoded_inputs)

    encoded = keras.layers.MaxPooling2D(
        pool_size=(2, 2),
        padding='same'
    )(encoded)

    for i in range(1, len(filters)):
        encoded = keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(encoded)

        encoded = keras.layers.MaxPooling2D(
            pool_size=(2, 2),
            padding='same'
        )(encoded)

    encoder_model = keras.Model(encoded_inputs, encoded)

    ##############################
    # DECODER MODEL
    ##############################
    decoded_inputs = keras.Input(shape=latent_dims)
    decoded = keras.layers.Conv2D(
        filters=filters[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='relu'
    )(decoded_inputs)

    decoded = keras.layers.UpSampling2D(
        size=(2, 2)
    )(decoded)

    for i in range(len(filters) - 1, 1, -1):
        decoded = keras.layers.Conv2D(
            filters=filters[i],
            kernel_size=(3, 3),
            padding='same',
            activation='relu'
        )(decoded)

        decoded = keras.layers.UpSampling2D(
            size=(2, 2)
        )(decoded)

    decoded = keras.layers.Conv2D(
        filters=filters[0],
        kernel_size=(3, 3),
        padding='valid',
        activation='relu'
    )(decoded)

    decoded = keras.layers.UpSampling2D(
        size=(2, 2)
    )(decoded)

    decoded = keras.layers.Conv2D(
        filters=input_dims[-1],
        kernel_size=(3, 3),
        padding='same',
        activation='sigmoid'
    )(decoded)

    decoder_model = keras.Model(decoded_inputs, decoded)

    ##############################
    # AUTOENCODER MODEL
    ##############################
    autoencoder_model = keras.Model(
        encoded_inputs,
        decoder_model(encoder_model(encoded_inputs))
    )

    autoencoder_model.compile(
        optimizer='adam',
        loss='binary_crossentropy'
    )

    return encoder_model, decoder_model, autoencoder_model
