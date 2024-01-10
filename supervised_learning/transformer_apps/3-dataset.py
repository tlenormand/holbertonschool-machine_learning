#!/usr/bin/env python3
""" Dataset """

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ Class that loads and preps a dataset for machine translation """
    def __init__(self, batch_size, max_len):
        """ Initializer """
        self.batch_size = batch_size
        self.max_len = max_len
        self.data_train = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="train",
            as_supervised=True
        )
        self.data_valid = tfds.load(
            "ted_hrlr_translate/pt_to_en",
            split="validation",
            as_supervised=True
        )
        self.tokenizer_pt, self.tokenizer_en = self.tokenize_dataset(
            self.data_train
        )
        self.data_train = self.data_train.map(self.tf_encode)
        self.data_valid = self.data_valid.map(self.tf_encode)

    def tokenize_dataset(self, data):
        """ Method that creates sub-word tokenizers for our dataset

        Args:
            data: (tf.data.Dataset) whose examples are formatted as a tuple
                  (pt, en)

        Returns:
            tokenizer_pt, tokenizer_en: (tfds.features.text.SubwordTextEncoder)
                the Portuguese tokenizer and the English tokenizer, respectively.
        """
        tokenizer_pt = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (pt.numpy() for pt, _ in data),
            target_vocab_size=2**15
        )
        tokenizer_en = tfds.deprecated.text.SubwordTextEncoder.build_from_corpus(
            (en.numpy() for _, en in data),
            target_vocab_size=2**15
        )

        return tokenizer_pt, tokenizer_en

    def encode(self, pt, en):
        """ Method that encodes a translation into tokens

        Args:
            pt: (tf.Tensor) containing the Portuguese sentence.
            en: (tf.Tensor) containing the corresponding English sentence.

        Returns:
            pt_tokens, en_tokens: (np.ndarray) containing the Portuguese tokens
                and the English tokens, respectively.
        """
        pt_start_index = self.tokenizer_pt.vocab_size
        pt_end_index = pt_start_index + 1
        en_start_index = self.tokenizer_en.vocab_size
        en_end_index = en_start_index + 1

        pt_tokens = [pt_start_index] + self.tokenizer_pt.encode(
            pt.numpy()) + [pt_end_index]
        en_tokens = [en_start_index] + self.tokenizer_en.encode(
            en.numpy()) + [en_end_index]

        return pt_tokens, en_tokens

    def tf_encode(self, pt, en):
        """ Method that acts as a tensorflow wrapper for the encode instance
            method

        Args:
            pt: (tf.Tensor) containing the Portuguese sentence.
            en: (tf.Tensor) containing the corresponding English sentence.

        Returns:
            pt_tokens, en_tokens: (tf.Tensor) containing the Portuguese tokens
                and the English tokens, respectively.
        """
        pt_encoded, en_encoded = tf.py_function(
            func=self.encode,
            inp=[pt, en],
            Tout=[tf.int64, tf.int64]
        )
        pt_encoded.set_shape([None])
        en_encoded.set_shape([None])

        return pt_encoded, en_encoded
