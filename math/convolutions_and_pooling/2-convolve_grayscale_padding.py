#!/usr/bin/env python3
""" Convolutional Neural Networks """

import numpy as np


def convolve_grayscale_padding(images, kernel, padding):
    """ performs a convolution on grayscale images with custom padding

    Arguments:
        images: a numpy.ndarray with shape (m, h, w) containing multiple
                grayscale images
            m: the number of images
            h: the height in pixels of the images
            w: the width in pixels of the images
        kernel: a numpy.ndarray with shape (kh, kw) containing the kernel
                for the convolution
            kh: the height of the kernel
            kw: the width of the kernel
        padding: a tuple of (ph, pw)
            ph: padding for the height of the image
            pw: padding for the width of the image

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
                (width, width)
            ),
            mode='constant'
        )

    nb_image, image_heigh, image_width = images.shape
    kernel_heigh, kernel_width = kernel.shape

    # add padding to the image in order to keep the same size
    images = add_padding(images, padding)

    # output size
    conv_heigh = image_heigh - kernel_heigh + 1 + 2 * padding[0]
    conv_width = image_width - kernel_width + 1 + 2 * padding[1]

    conv = np.zeros((nb_image, conv_heigh, conv_width))
    for heigh in range(conv_heigh):
        for width in range(conv_width):
            # every image
            # height from i to kernel height + i
            # width from j to kernel width + j
            image = images[
                :,
                heigh:heigh + kernel_heigh,
                width:width + kernel_width
            ]
            # kernel contains operations for every image
            # axis if for the image
            # 0 the number of images, 1 the height, 2 the width
            conv[:, heigh, width] = np.sum(image * kernel, axis=(1, 2))

    return conv
