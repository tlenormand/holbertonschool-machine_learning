#!/usr/bin/env python3
""" Dataset """

import tensorflow.compat.v2 as tf
import tensorflow_datasets as tfds


class Dataset:
    """ Class that loads and preps a dataset for machine translation """
    def __init__(self):
        """ Initializer """
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
