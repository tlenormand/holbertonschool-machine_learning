#!/usr/bin/env python3
""" Pooling """

import numpy as np


def pool(images, kernel_shape, stride, mode='max'):
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
        stride: a tuple of (sh, sw)
            sh: stride for the height of the image
            sw: stride for the width of the image
        mode: indicates the type of pooling
            max: max pooling
            avg: average pooling

    Returns:
        a numpy.ndarray containing the convolved images
    """
    nb_image, image_heigh, image_width, image_channels = images.shape
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
                    images[
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
                    images[
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
