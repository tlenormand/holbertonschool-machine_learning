#!/usr/bin/env python3
""" Sparse Autoencoder """

import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims, lambtha):
    """ Function that creates a sparse autoencoder

    Arguments:
        input_dims {int} -- Is the input size
        hidden_layers {list} -- Is the list containing the number of nodes
            for each hidden layer in the encoder, respectively
        latent_dims {int} -- Is the latent space dimensionality
        lambtha {float} -- Is the regularization parameter used for L1
            regularization on the encoded output

    Returns:
        tuple -- Containing the encoder, decoder and autoencoder models
    """
    ##############################
    # ENCODER MODEL
    ##############################
    encoded_inputs = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        hidden_layers[0],
        activation='relu'
    )(encoded_inputs)

    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(
            hidden_layers[i],
            activation='relu'
        )(encoded)

    encoded = keras.layers.Dense(
        latent_dims,
        activation='relu',
        activity_regularizer=keras.regularizers.l1(lambtha)
    )(encoded)

    encoder_model = keras.Model(encoded_inputs, encoded)

    ##############################
    # DECODER MODEL
    ##############################
    decoded_inputs = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(
        hidden_layers[-1],
        activation='relu'
    )(decoded_inputs)

    for i in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(
            hidden_layers[i],
            activation='relu'
        )(decoded)

    decoded = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(decoded)

    decoder_model = keras.Model(decoded_inputs, decoded)

    ##############################
    # AUTOENCODER MODEL
    ##############################
    auto_model = keras.Model(
        encoded_inputs,
        decoder_model(encoder_model(encoded_inputs))
    )

    auto_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, auto_model
