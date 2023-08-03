#!/usr/bin/env python3
"""
module containing function that create a variational autoencoder
"""
import tensorflow.keras as keras


def sampling(args):
    """
    function that sampling from an isotropic unit Gaussian
    Args:
        args: tensor represent mean and log variance of Q(z|X)
    Return:
        z: tensor represent sampled latent_vector
    """
    z_mean, z_log_variance = args
    batch = keras.backend.shape(z_mean)[0]
    dim = keras.backend.int_shape(z_mean)[1]
    epsilon = keras.backend.random_normal(shape=(batch, dim))
    return z_mean + keras.backend.exp(0.5 * z_log_variance) * epsilon


def autoencoder(input_dims, hidden_layers, latent_dims):
    """
    function that creates a variational autoencoder
    Args:
        input_dims: integer containing the dimensions of the model input
        hidden_layers: list containing the number of nodes for each hidden
            layer in the encoder, decoder
        latent_dims: integer containing the dimensions of the latent space
            representation
    Return: encoder, decoder, auto
        encoder: the encoder model, output the latent representation, the mean,
            and the log variance
        decoder: the decoder model
        auto: the full autoencoder model
    """
    encoder_inputs = keras.Input(shape=(input_dims,))

    # encoder model
    for i in range(len(hidden_layers)):
        if i == 0:
            encoded_output = keras.layers.Dense(
                hidden_layers[i],
                activation='relu'
            )(encoder_inputs)
        else:
            encoded_output = keras.layers.Dense(
                hidden_layers[i],
                activation='relu'
            )(encoded_output)
    z_mean = keras.layers.Dense(
        latent_dims
    )(encoded_output)
    z_log_variance = keras.layers.Dense(
        latent_dims
    )(encoded_output)
    z = keras.layers.Lambda(sampling)([z_mean, z_log_variance])

    encoder_model = keras.Model(inputs=encoder_inputs,
                                outputs=[z, z_mean, z_log_variance])

    # decoder model
    decoder_inputs = keras.Input(shape=(latent_dims,))
    for i in range(len(hidden_layers) - 1, -1, -1):
        if i == len(hidden_layers) - 1:
            decoded_output = keras.layers.Dense(
                hidden_layers[i],
                activation='relu'
            )(decoder_inputs)
        else:
            decoded_output = keras.layers.Dense(
                hidden_layers[i],
                activation='relu'
            )(decoded_output)
    decoded_output = keras.layers.Dense(
        input_dims,
        activation='sigmoid'
    )(decoded_output)

    decoder_model = keras.Model(inputs=decoder_inputs, outputs=decoded_output)

    # full autoencoder model
    autoencoder_model = keras.Model(
        inputs=encoder_inputs,
        outputs=decoder_model(encoder_model(encoder_inputs)))

    # loss reconstruction
    reconstruction_loss = keras.losses.binary_crossentropy(
        encoder_inputs,
        encoder_model(encoder_inputs)
    )
    reconstruction_loss *= input_dims
    exponent = keras.backend.exp(z_log_variance)
    kl_loss = 1 + z_log_variance - keras.backend.square(z_mean) - exponent
    kl_loss = keras.backend.sum(kl_loss, axis=-1)
    kl_loss *= -0.5
    vae_loss = keras.backend.mean(reconstruction_loss + kl_loss)

    autoencoder_model.compile(optimizer='adam',
                              loss=vae_loss)

    return encoder_model, decoder_model, autoencoder_model