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
    
    height_A_prev, width_A_prev, channel_A_prev, nb_A_prev = A_prev.shape
    height_dz, width_dz, channel_dz, nb_dz = dZ.shape
    height_kernel, width_kernel, channel_kernel, nb_kernel = W.shape
    stride_heigh, stride_width = stride
    
    # add padding to the image in order to keep the same size
    if padding == 'valid':
        padding = (0, 0)
    elif padding == 'same':
        padding = (
            int(((height_A_prev - 1) * stride_heigh +
                height_kernel - height_A_prev) / 2 + 1),
            int(((width_A_prev - 1) * stride_width +
                width_kernel - width_A_prev) / 2 + 1)
        )

    A_prev = add_padding(A_prev, padding)

    padding_heigh, padding_width = padding

    # output size
    output_heigh = int(1 + (height_A_prev + 2 * padding_heigh - height_kernel) / stride_heigh)
    output_width = int(1 + (width_A_prev + 2 * padding_width - width_kernel) / stride_width)

    # convolution output
    dA_prev = np.zeros(A_prev.shape)
    dW = np.zeros(W.shape)
    db = np.zeros(b.shape)

    for i in range(nb_A_prev):
        for j in range(output_heigh):
            for k in range(output_width):
                for l in range(nb_kernel):

    return dA_prev, dW, db
