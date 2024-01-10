#!/usr/bin/env python3
"""	Create Masks """

import tensorflow.compat.v2 as tf


def create_masks(inputs, target):
    """ creates all masks for training/validation

    Args:
        inputs: tf.Tensor (batch, seq_len_in) of input tokens
        target: tf.Tensor (batch, seq_len_out) of target tokens
    
    Returns:
        encoder_mask: tf.Tensor padding mask shape (batch, 1, 1, seq_len_in)
            1 in position that should be masked, 0 in pad
        look_ahead_mask: tf.Tensor look ahead mask shape (batch, 1, seq_out, seq_out)
            1 in pos to be masked, 0 in pos not to be masked
        decoder_mask: tf.Tensor padding mask shape (batch, 1, 1, seq_len_in)
            1 in pos that should be masked, 0 in pad
    """
    _, seq_len_out = target.shape

    # encoder mask
    encoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    encoder_mask = encoder_mask[:, tf.newaxis, tf.newaxis, :]

    # look ahead mask
    look_ahead_mask = 1 - tf.linalg.band_part(tf.ones((seq_len_out, seq_len_out)), -1, 0)
    look_ahead_mask = look_ahead_mask[tf.newaxis, tf.newaxis, :, :]

    # decoder mask
    decoder_mask = tf.cast(tf.math.equal(inputs, 0), tf.float32)
    decoder_mask = decoder_mask[:, tf.newaxis, tf.newaxis, :]

    return encoder_mask, look_ahead_mask, decoder_mask
