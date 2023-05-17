#!/usr/bin/env python3
""" Pooling Back Prop """

import numpy as np


def pool_backward(dA, A_prev, kernel_shape, stride=(1, 1), mode='max'):
    """ performs back propagation over a pooling layer of a neural network

    Arguments:
        dA: numpy.ndarray of shape (m, h_new, w_new, c_new) containing the
                partial derivatives with respect to the output of the pooling
                layer
            h_new: the height of the output
            w_new: the width of the output
        A_prev: numpy.ndarray of shape (m, h_prev, w_prev, c) containing the
                output of the previous layer
            h_prev: the height of the previous layer
            w_prev: the width of the previous layer
        kernel_shape: tuple of (kh, kw) containing the size of the kernel for
                the pooling
            kh: the kernel height
            kw: the kernel width
        stride: tuple of (sh, sw) containing the strides for the pooling
            sh: the stride for the height
            sw: the stride for the width
        mode: string containing either max or avg, indicating whether to
                perform maximum or average pooling, respectively

    Returns:
        the partial derivatives with respect to the previous layer (dA_prev)
    """
    number_dA, height_dA, width_dA, channel_dA = dA.shape
    nb_A_prev, height_A_prev, width_A_prev, channel_A_prev = A_prev.shape
    height_kernel, width_kernel = kernel_shape
    stride_heigh, stride_width = stride

    # initialize the output
    dA_prev = np.zeros(A_prev.shape)

    for n in range(number_dA):
        for h in range(height_dA):
            for w in range(width_dA):
                for c in range(channel_dA):
                    h_start = h * stride_heigh
                    h_end = h_start + height_kernel
                    w_start = w * stride_width
                    w_end = w_start + width_kernel

                    # slice A_prev
                    slice_A_prev = A_prev[n, h_start:h_end, w_start:w_end, c]

                    # compute the gradient
                    if mode == 'max':
                        mask = (slice_A_prev == np.max(slice_A_prev))
                        dA_prev[n, h_start:h_end, w_start:w_end, c] += (
                            mask * dA[n, h, w, c]
                        )
                    elif mode == 'avg':
                        average = (dA[n, h, w, c] /
                                   (height_kernel * width_kernel))
                        dA_prev[n, h_start:h_end, w_start:w_end, c] += (
                            np.ones(kernel_shape) * average
                        )

    return dA_prev
