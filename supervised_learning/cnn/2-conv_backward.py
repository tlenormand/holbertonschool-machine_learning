#!/usr/bin/env python3
""" Convolutional Back Prop """

import numpy as np


def conv_backward(dZ, A_prev, W, b, padding="same", stride=(1, 1)):
    """ performs back propagation over a convolutional layer of a neural
            network

    Arguments:
        dZ: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
                partial derivatives with respect to the unactivated output of
                the convolutional layer
            h_new: the height of the output
            w_new: the width of the output
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) containing
                the output of the previous layer
            h_prev: the height of the previous layer
            w_prev: the width of the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) containing the
                kernels for the convolution
            kh: the filter height
            kw: the filter width
        b: numpy.ndarray of shape (1, 1, 1, c_new) containing the biases
                applied to the convolution
        padding: string that is either same or valid, indicating the type of
                padding used
        stride: tuple of (sh, sw) containing the strides for the convolution
            sh: the stride for the height
            sw: the stride for the width

    Returns:
        the partial derivatives with respect to the previous layer (dA_prev),
            the kernels (dW), and the biases (db), respectively
    """
    m, h_prev, w_prev, ch_prev = A_prev.shape
    h_stride, w_stride = stride
    kh, kw, _, _ = W.shape
    _, h_new, w_new, c_new = dZ.shape

    if padding == "same":
        h_pad = int((((h_prev - 1) * h_stride + kh - h_prev) / 2) + 1)
        w_pad = int((((w_prev - 1) * w_stride + kw - w_prev) / 2) + 1)
    else:
        h_pad, w_pad = 0, 0

    A_prev_padded = np.pad(A_prev, [(0, 0), (h_pad, h_pad), (w_pad, w_pad), (0, 0)], mode="constant")
    dA_padded = np.zeros(shape=A_prev_padded.shape)
    dW = np.zeros(shape=W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    for sample in range(m):
        for h_i in range(h_new):
            for w_i in range(w_new):
                for ch in range(c_new):
                    h_is = h_i * h_stride
                    w_is = w_i * w_stride
                    dA_padded[sample, h_is:h_is + kh, w_is:w_is + kw, :] += W[:, :, :, ch] * dZ[sample, h_i, w_i, ch]
                    dW[:, :, :, ch] += A_prev_padded[sample, h_is:h_is + kh, w_is:w_is + kw, :] * dZ[sample, h_i, w_i, ch]

    if padding == "same":
        dA = dA_padded[:, h_pad:h_prev - h_pad, w_pad:w_prev - w_pad, :]
    else:
        dA = dA_padded

    return dA, dW, db
