#!/usr/bin/env python3
""" None """

import tensorflow as tf


def flip_image(image):
    """ None """
    return tf.image.flip_left_right(image)
