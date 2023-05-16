#!/usr/bin/env python3
""" Convolutional Forward Prop"""

import numpy as np


def conv_forward(A_prev, W, b, activation, padding="same", stride=(1, 1)):
    """ performs forward propagation over a convolutional layer of a neural
            network

    Arguments:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) with the
                output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        W: numpy.ndarray of shape (kh, kw, c_prev, c_new) with the kernels for
                the convolution
            kh: filter height
            kw: filter width
            c_prev: number of channels in the previous layer
            c_new: number of channels in the output
        b: numpy.ndarray of shape (1, 1, 1, c_new) with the biases applied to
                the convolution
        activation: activation function applied to the convolution
        padding: string that is either same or valid, indicating the type of
                padding used
        stride: tuple of (sh, sw) containing the strides for the convolution
            sh: stride for the height
            sw: stride for the width

    Returns:
        the output of the convolutional layer
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

    nb_image, image_heigh, image_width, image_channels = A_prev.shape
    kernel_heigh, kernel_width, kernel_channels, nb_kernel = W.shape
    stride_heigh, stride_width = stride

    # add padding to the image in order to keep the same size
    if padding == 'valid':
        padding = (0, 0)
    elif padding == 'same':
        padding = (
            int(((image_heigh - 1) * stride_heigh +
                kernel_heigh - image_heigh) / 2 + 1),
            int(((image_width - 1) * stride_width +
                kernel_width - image_width) / 2 + 1)
        )

    A_prev = add_padding(A_prev, padding)

    # output size
    output_heigh = int((image_heigh + 2 * padding[0] - kernel_heigh) /
                       stride_heigh + 1)
    output_width = int((image_width + 2 * padding[1] - kernel_width) /
                       stride_width + 1)

    # convolution output
    output = np.zeros((nb_image, output_heigh, output_width, nb_kernel))

    images = np.arange(nb_image)
    for i in range(output_heigh):
        for j in range(output_width):
            for k in range(nb_kernel):
                output[images, i, j, k] = np.sum(
                    A_prev[
                        images,
                        i * stride_heigh: i * stride_heigh + kernel_heigh,
                        j * stride_width: j * stride_width + kernel_width
                    ] * W[..., k],  # ... same as W[:, :, :, k]
                    axis=(1, 2, 3)
                )

    return activation(output + b)
