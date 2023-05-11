#!/usr/bin/env python3
""" Convolutional Neural Networks """

import numpy as np


def convolve_grayscale_same(images, kernel):
    """ performs a valid convolution on grayscale images

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

    Returns:
        a numpy.ndarray containing the convolved images
    """
    nb_image, image_heigh, image_width = images.shape
    kernel_heigh, kernel_width = kernel.shape

    # output size
    output_heigh = int(image_heigh)
    output_width = int(image_width)

    # convolution output
    conv = np.zeros((nb_image, output_heigh, output_width))

    # Add padding to the images
    pad_height = int(kernel_heigh / 2)
    pad_width = int(kernel_width / 2)
    images = np.pad(
        images,
        # ((before height, after height), (before width, after width))
        pad_width=((0, 0), (pad_height, pad_height), (pad_width, pad_width)),
        mode='constant'
    )

    for height in range(output_heigh):
        for width in range(output_width):
            conv[:, height, width] = np.sum(
                images[
                    :,
                    height: height + kernel_heigh,
                    width: width + kernel_width
                ] * kernel,
                axis=(1, 2)
            )

    return conv
