#!/usr/bin/env python3
""" 11. Transformer Network """

import tensorflow as tf

Encoder = __import__('9-transformer_encoder').Encoder
Decoder = __import__('10-transformer_decoder').Decoder


class Transformer(tf.keras.Model):
    """ Class that instantiates a Transformer """
    def __init__(self, N, dm, h, hidden, input_vocab, target_vocab,
                 max_seq_input, max_seq_target, drop_rate=0.1):
        """ Initializer

        Args:
            N: (int) the number of blocks in the encoder.
            dm: (int) the dimensionality of the model.
            h: (int) the number of heads.
            hidden: (int) the number of hidden units in the fully connected
                    layer.
            input_vocab: (int) the size of the input vocabulary.
            target_vocab: (int) the size of the target vocabulary.
            max_seq_input: (int) the maximum sequence length possible for the
                           input.
            max_seq_target: (int) the maximum sequence length possible for the
                            target.
            drop_rate: (float) the dropout rate.
        """
        super(Transformer, self).__init__()
        self.encoder = Encoder(
            N,
            dm,
            h,
            hidden,
            input_vocab,
            max_seq_input,
            drop_rate
        )
        self.decoder = Decoder(
            N,
            dm,
            h,
            hidden,
            target_vocab,
            max_seq_target,
            drop_rate
        )
        self.linear = tf.keras.layers.Dense(
            units=target_vocab,
            activation='softmax'
        )

    def call(self, inputs, target, training, encoder_mask, look_ahead_mask,
             decoder_mask):
        """ call method

        Args:
            inputs: (tf.Tensor) containing the inputs.
            target: (tf.Tensor) containing the target.
            training: (bool) to determine if the model is training.
            encoder_mask: the padding mask to be applied to the encoder.
            look_ahead_mask: the look ahead mask to be applied to the decoder.
            decoder_mask: the padding mask to be applied to the decoder.

        Returns:
            (tf.Tensor) containing the transformer output.
        """
        enc_output = self.encoder(inputs, training, encoder_mask)
        dec_output = self.decoder(
            target,
            enc_output,
            training,
            look_ahead_mask,
            decoder_mask
        )
        final_output = self.linear(dec_output)

        return final_output
