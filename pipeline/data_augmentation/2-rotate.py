#!/usr/bin/env python3
""" None """

import tensorflow as tf


def rotate_image(image):
    """ None """
    return tf.image.rot90(image, k=1)
