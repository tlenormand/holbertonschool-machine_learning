#!/usr/bin/env python3
""" Pooling Forward Prop """

import numpy as np


def pool_forward(A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs forward propagation over a pooling layer of a neural network

    Arguments:
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c_prev) with the
                output of the previous layer
            m: number of examples
            h_prev: height of the previous layer
            w_prev: width of the previous layer
            c_prev: number of channels in the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the kernel for
                the pooling
            kh: kernel height
            kw: kernel width
        stride: tuple of (sh, sw) containing the strides for the pooling
            sh: stride for the height
            sw: stride for the width
        mode: string containing either max or avg, indicating whether to
                perform maximum or average pooling, respectively

    Returns:
        the output of the pooling layer
    """
    nb_image, image_heigh, image_width, image_channels = A_prev.shape
    kernel_heigh, kernel_width = kernel_shape
    stride_heigh, stride_width = stride

    # output size
    output_heigh = int((image_heigh - kernel_heigh) / stride_heigh + 1)
    output_width = int((image_width - kernel_width) / stride_width + 1)

    # convolution output
    conv = np.zeros((nb_image, output_heigh, output_width, image_channels))

    for height in range(output_heigh):
        for width in range(output_width):
            if mode == 'max':
                conv[:, height, width, :] = np.max(
                    A_prev[
                        :,
                        height * stride_heigh: height *
                        stride_heigh + kernel_heigh,
                        width * stride_width: width *
                        stride_width + kernel_width,
                        :
                    ],
                    axis=(1, 2)
                )
            elif mode == 'avg':
                conv[:, height, width, :] = np.mean(
                    A_prev[
                        :,
                        height * stride_heigh: height *
                        stride_heigh + kernel_heigh,
                        width * stride_width: width *
                        stride_width + kernel_width,
                        :
                    ],
                    axis=(1, 2)
                )

    return conv
