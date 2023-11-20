#!/usr/bin/env python3
""" None """

import tensorflow as tf


def change_hue(image, delta):
    """ None """
    return tf.image.adjust_hue(image, delta)
