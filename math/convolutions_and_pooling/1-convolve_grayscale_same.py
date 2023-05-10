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
    def add_padding(images):
        """ add padding to the image """
        if kernel_heigh % 2:
            padding_heigh = int((kernel_heigh - 1) / 2)
        else:
            padding_heigh = int(kernel_heigh / 2)

        if kernel_width % 2:
            padding_width = int((kernel_width - 1) / 2)
        else:
            padding_width = int(kernel_width / 2)

        return np.pad(
            images,
            # ((before height, after height), (before width, after width))
            pad_width=(
                (0, 0),
                (padding_heigh, padding_heigh),
                (padding_width, padding_width)
            ),
            mode='constant'
        )

    # add padding to the image in order to keep the same size
    images = add_padding(images)

    nb_image, image_heigh, image_width = images.shape
    kernel_heigh, kernel_width = kernel.shape

    # output size
    conv_heigh = image_heigh - kernel_heigh + 1
    conv_width = image_width - kernel_width + 1

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
