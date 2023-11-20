#!/usr/bin/env python3
""" None """

import tensorflow as tf


def shear_image(image, intensity):
    """ None """
    return tf.keras.preprocessing.image.random_shear(image, intensity)
