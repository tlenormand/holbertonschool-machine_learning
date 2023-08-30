#!/usr/bin/env python3
""" RNN Decoder """

import tensorflow as tf

SelfAttention = __import__('1-self_attention').SelfAttention


class RNNDecoder(tf.keras.layers.Layer):
    """ class RNNDecoder """
    def __init__(self, vocab, embedding, units, batch):
        """ Initializer

        Args:
            vocab: (int) the size of the input vocabulary
            embedding: (int) the dimensionality of the embedding vector
            units: (int) the number of hidden units in the RNN cell
            batch: (int) the batch size
        """
        super().__init__()
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.F = tf.keras.layers.Dense(vocab)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer="glorot_uniform",
            return_sequences=True,
            return_state=True
        )

    def call(self, x, s_prev, hidden_states):
        """ call method

        Args:
            x: (tf.Tensor) containing the previous word in the target sequence
                as an index of the target vocabulary
            s_prev: (tf.Tensor) containing the previous decoder hidden state
            hidden_states: (tf.Tensor) containing the outputs of the encoder

        Returns:
            y, s
            y: (tf.Tensor) containing the output word as a one hot vector in
               the target vocabulary
            s: (tf.Tensor) containing the new decoder hidden state
        """
        context, _ = SelfAttention(s_prev.shape[1])(s_prev, hidden_states)
        x = self.embedding(x)
        x = tf.concat([tf.expand_dims(context, axis=1), x], axis=-1)
        y, s = self.gru(x)
        y = tf.reshape(y, (-1, y.shape[2]))
        y = self.F(y)

        return y, s
