#!/usr/bin/env python3
""" None """

import tensorflow as tf


def crop_image(image, size):
    """ None """
    return tf.image.random_crop(image, size)
