#!/usr/bin/env python3
""" Positional Encoding """

import numpy as np


def positional_encoding(max_seq_len, dm):
    """ calculates the positional encoding for a transformer

    Args:
        max_seq_len: (int) the maximum sequence length
        dm: (int) the model depth

    Returns:
        (numpy.ndarray) containing the positional encoding vectors
    """
    pos = np.arange(max_seq_len)[:, np.newaxis]
    i = np.arange(dm)[np.newaxis, :]
    PE = pos / np.power(10000, (2 * (i // 2)) / np.float32(dm))
    PE[:, 0::2] = np.sin(PE[:, 0::2])
    PE[:, 1::2] = np.cos(PE[:, 1::2])

    return PE
