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
    def add_padding(images, padding):
        """ add padding to the image """
        height, width = padding

        return np.pad(
            images,
            # ((before height, after height), (before width, after width))
            pad_width=(
                (0, 0),
                (height, height),
                (width, width),
                (0, 0)
            ),
            mode='constant'
        )

    nb_A_prev, height_A_prev, width_A_prev, channel_A_prev = A_prev.shape
    number_dz, height_dz, width_dz, channel_dz = dZ.shape
    height_kernel, width_kernel, prev_kernel, nb_kernel = W.shape
    stride_heigh, stride_width = stride

    # convolution output
    dA = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.sum(dZ, axis=(0, 1, 2), keepdims=True)

    # add padding to the image in order to keep the same size
    if padding == 'valid':
        padding = (0, 0)
    elif padding == 'same':
        padding = (
            int(((height_A_prev - 1) * stride_heigh +
                height_kernel - height_A_prev) // 2),
            int(((width_A_prev - 1) * stride_width +
                width_kernel - width_A_prev) // 2)
        )

    dA = add_padding(A_prev, padding)

    for n in range(number_dz):  # n number of images
        for h in range(height_dz):  # h height of the output
            for w in range(width_dz):  # w width of the output
                for k in range(channel_dz):  # k number of kernels
                    mat = dA[
                        n,
                        h * stride_heigh: h * stride_heigh + height_kernel,
                        w * stride_width: w * stride_width + width_kernel,
                        :
                    ]
                    dA[
                        n,
                        h * stride_heigh: h * stride_heigh + height_kernel,
                        w * stride_width: w * stride_width + width_kernel,
                        :
                    ] += W[..., k] * dZ[n, h, w, k]
                    dW[..., k] += mat * dZ[n, h, w, k]

    return dA, dW, db
