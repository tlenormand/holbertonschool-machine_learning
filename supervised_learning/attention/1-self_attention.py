#!/usr/bin/env python3
""" Self Attention """

import tensorflow as tf


class SelfAttention(tf.keras.layers.Layer):
    """ class SelfAttention """
    def __init__(self, units):
        """ Initializer

        Args:
            units: (int) the number of hidden units in the alignment model.
        """
        super().__init__()
        self.units = units
        self.W = tf.keras.layers.Dense(units)
        self.U = tf.keras.layers.Dense(units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, s_prev, hidden_states):
        """ call method

        Args:
            s_prev: (tf.Tensor) containing the previous decoder hidden state.
            hidden_states: (tf.Tensor) containing the outputs of the encoder.

        Returns:
            context, weights.
            context: (tf.Tensor) containing the context vector for the decoder.
            weights: (tf.Tensor) containing the attention weights.
        """
        s_prev = tf.expand_dims(s_prev, axis=1)
        score = self.V(tf.nn.tanh(self.W(s_prev) + self.U(hidden_states)))
        weights = tf.nn.softmax(score, axis=1)
        context = weights * hidden_states
        context = tf.reduce_sum(context, axis=1)

        return context, weights