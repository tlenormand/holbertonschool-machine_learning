#!/usr/bin/env python3
""" Scaled Dot Product Attention """

import tensorflow as tf


def sdp_attention(Q, K, V, mask=None):
    """ calculates the scaled dot product attention

    Args:
        Q: (tf.Tensor) containing the query matrix
        K: (tf.Tensor) containing the key matrix
        V: (tf.Tensor) containing the value matrix
        mask: (tf.Tensor) containing the optional mask, or defaulted to None

    Returns:
        output: (tf.Tensor) containing the scaled dot product attention
        weights: (tf.Tensor) containing the attention weights
    """
    # matmul
    matmul_qk = tf.matmul(Q, K, transpose_b=True)
    # scale
    dk = tf.cast(tf.shape(K)[-1], tf.float32)
    scaled_attention_logits = matmul_qk / tf.math.sqrt(dk)

    # mask
    if mask is not None:
        scaled_attention_logits += (mask * -1e9)

    # softmax
    weights = tf.nn.softmax(scaled_attention_logits, axis=-1)
    # matmul
    output = tf.matmul(weights, V)

    return output, weights
