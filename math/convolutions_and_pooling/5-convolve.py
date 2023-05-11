#!/usr/bin/env python3
""" Convolution with Channels """

import numpy as np


def convolve(images, kernels, padding='same', stride=(1, 1)):
    """ performs a convolution on grayscale images

    Arguments:
        images: a numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
            m: the number of images
            h: the height in pixels of the images
            w: the width in pixels of the images
            c: the number of channels in the image
        kernel: a numpy.ndarray with shape (kh, kw) containing the kernel
                for the convolution
            kh: the height of the kernel
            kw: the width of the kernel
            c: the number of channels in the kernel
            nc: the number of kernels
        padding: a tuple of (ph, pw)
            ph: padding for the height of the image
            pw: padding for the width of the image
        stride: a tuple of (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image

    Returns:
        a numpy.ndarray containing the convolved images
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

    nb_image, image_heigh, image_width, image_channels = images.shape
    kernel_heigh, kernel_width, kernel_channels, nb_kernel = kernels.shape
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

    images = add_padding(images, padding)

    # output size
    conv_heigh = int(
        (image_heigh - kernel_heigh + 2 * padding[0]) / stride_heigh
    ) + 1
    conv_width = int(
        (image_width - kernel_width + 2 * padding[1]) / stride_width
    ) + 1

    conv = np.zeros((nb_image, conv_heigh, conv_width, nb_kernel))

    for i_kernel in range(nb_kernel):
        for heigh in range(conv_heigh):
            for width in range(conv_width):
                heigh_strided = heigh * stride_heigh
                width_strided = width * stride_width
                image = images[
                    :,
                    heigh_strided:heigh_strided + kernel_heigh,
                    width_strided:width_strided + kernel_width
                ]
                kernel = kernels[:, :, :, i_kernel]
                conv[:, heigh, width, i_kernel] = np.sum(
                    image * kernel,
                    axis=(1, 2, 3)
                )

    return conv
