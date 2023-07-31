#!/usr/bin/env python3
"""
module containing function that create a vanilla autoencoder
"""
import tensorflow.keras as keras


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    function that creates a vanilla autoencoder
    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
            layer in the encoder
        latent_dims: integer containing the dimensions of the latent space
            representation
    Return: encoder, decoder, auto
        encoder: the encoder model
        decoder: the decoder model
        auto: the full autoencoder model
    """
    ##############################
    # ENCODER MODEL
    ##############################
    encoded_inputs = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(
        hidden_layers[0],
        activation='relu'
    )(encoded_inputs)

    for i in range(len(hidden_layers) - 1):
        encoded = keras.layers.Dense(
            hidden_layers[i],
            activation='relu'
        )(encoded)

    encoded_output = keras.layers.Dense(
        latent_dims,
        activation='relu'
    )(encoded)

    encoder_model = keras.Model(encoded_inputs, encoded_output)

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

    decoded_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(decoded)

    decoder_model = keras.Model(decoded_inputs, decoded_output)

    ##############################
    # AUTOENCODER MODEL
    ##############################
    auto_input = keras.Input(shape=(input_dims,))
    auto_model = keras.Model(
        auto_input,
        decoder_model(encoder_model(auto_input)))

    auto_model.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder_model, decoder_model, auto_model
