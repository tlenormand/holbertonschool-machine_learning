#!/usr/bin/env python3
""" RNN Encoder """

import tensorflow as tf


class RNNEncoder(tf.keras.layers.Layer):
    """ class RNNEncoder """
    def __init__(self, vocab, embedding, units, batch):
        """ Initializer

        Args:
            vocab: (int) the size of the input vocabulary.
            embedding: (int) the dimensionality of the embedding vector.
            units: (int) the number of hidden units in the RNN cell.
            batch: (int) the batch size.
        """
        super().__init__()
        self.batch = batch
        self.units = units
        self.embedding = tf.keras.layers.Embedding(vocab, embedding)
        self.gru = tf.keras.layers.GRU(
            units,
            recurrent_initializer="glorot_uniform",
            return_sequences=True,
            return_state=True
        )

    def initialize_hidden_state(self):
        """ Initializes the hidden states for the RNN cell to a tensor of zeros

        Returns:
            A tensor of shape (batch, units)containing the initialized
            hidden states.
        """
        initializer = tf.keras.initializers.Zeros()
        tensor = initializer(shape=(self.batch, self.units))

        return tensor

    def call(self, x, initial):
        """ call method

        Args:
            x: (tf.Tensor) containing the input to the encoder layer as a
                tensor of shape (batch, input_seq_len).
            initial: (tf.Tensor) containing the initial hidden state as a
                        tensor of shape (batch, units).

        Returns:
            outputs, hidden.
            outputs: (tf.Tensor) containing the outputs of the encoder as a
                        tensor of shape (batch, input_seq_len, units).
            hidden: (tf.Tensor) containing the last hidden state of the
                    encoder as a tensor of shape (batch, units).
        """
        x = self.embedding(x)
        outputs, hidden = self.gru(x, initial_state=initial)

        return outputs, hidden
