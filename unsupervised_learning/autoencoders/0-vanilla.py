#!/usr/bin/env python3

import tensorflow.keras as keras


def encoder_block(input_dims, hidden_layers, latent_dims):
    """Creates an encoder block for a variational autoencoder
    Args:
        input_dims (int): the dimensions of the model input
        hidden_layers (list): list containing the number of nodes for each
            hidden layer in the encoder, respectively
        latent_dims (int): the dimensions of the latent space representation
    Returns:
        encoder, mu, sigma
            encoder is the encoder model
            mu is the model's mean output
            sigma is the model's standard deviation output
    """
    inputs = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)

    # iterate through hidden layers
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(
            hidden_layers[i],
            activation='relu'
        )(encoded)

    # layer for latent space representation
    encoded = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    encoder = keras.Model(inputs, encoded)

    return encoder

def decoder_block(input_dims, hidden_layers, latent_dims):
    """Creates a decoder block for a variational autoencoder
    Args:
        input_dims (int): the dimensions of the model input
        hidden_layers (list): list containing the number of nodes for each
            hidden layer in the decoder, respectively
        latent_dims (int): the dimensions of the latent space representation
    Returns:
        decoder, decoded
            decoder is the decoder model
            decoded is the decoded output
    """
    inputs = keras.Input(shape=(latent_dims,))
    decoded = keras.layers.Dense(hidden_layers[-1], activation='relu')(inputs)

    # iterate through hidden layers
    for i in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(
            hidden_layers[i],
            activation='relu'
        )(decoded)

    # output layer = input layer (dimensions should match)
    decoded = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    decoder = keras.Model(inputs, decoded)

    return decoder

def autoencoder_block(input_dims, hidden_layers, latent_dims):
    """Creates the full variational autoencoder
    Args:
        input_dims (int): the dimensions of the model input
        hidden_layers (list): list containing the number of nodes for each
            hidden layer in the encoder, respectively
        latent_dims (int): the dimensions of the latent space representation
    Returns:
        encoder, decoder, auto
            encoder is the encoder model
            decoder is the decoder model
            auto is the full autoencoder model
    """
    inputs = keras.Input(shape=(input_dims,))
    encoded = keras.layers.Dense(hidden_layers[0], activation='relu')(inputs)

    # iterate through hidden layers
    for i in range(1, len(hidden_layers)):
        encoded = keras.layers.Dense(
            hidden_layers[i],
            activation='relu'
        )(encoded)

    # layer for latent space representation
    encoded = keras.layers.Dense(latent_dims, activation='relu')(encoded)

    decoded = keras.layers.Dense(hidden_layers[-1], activation='relu')(encoded)

    # iterate through hidden layers
    for i in range(len(hidden_layers) - 2, -1, -1):
        decoded = keras.layers.Dense(
            hidden_layers[i],
            activation='relu'
        )(decoded)

    # output layer = input layer (dimensions should match)
    auto = keras.layers.Dense(input_dims, activation='sigmoid')(decoded)

    auto = keras.Model(inputs, auto)

    return auto

def autoencoder(input_dims, hidden_layers, latent_dims):
    """Creates an autoencoder
    Args:
        input_dims (int): the input size
        hidden_layers (list): contains the number of nodes for each hidden layer
        latent_dims (int): the size of the latent space representation
    Returns:
        encoder, decoder, auto
            encoder is the encoder model
            decoder is the decoder model
            auto is the full autoencoder model
    """
    encoder = encoder_block(input_dims, hidden_layers, latent_dims)
    decoder = decoder_block(input_dims, hidden_layers, latent_dims)
    auto = autoencoder_block(input_dims, hidden_layers, latent_dims)
    
    auto.compile(optimizer='adam', loss='binary_crossentropy')

    return encoder, decoder, auto
