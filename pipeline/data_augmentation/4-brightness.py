#!/usr/bin/env python3
""" None """

import tensorflow as tf


def change_brightness(image, max_delta):
    """ None """
    return tf.image.random_brightness(image, max_delta)
