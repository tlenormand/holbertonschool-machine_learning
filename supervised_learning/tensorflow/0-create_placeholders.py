#!/usr/bin/env python3
"""
module containing function create_placeholders
"""
import tensorflow as tf


def create_placeholders(nx, classes):
    """
    function that returns two placeholders, x and y, for the neural network
    Args:
        nx: the number of feature columns in our data
        classes: the number of classes in our classifier
    Return: two placeholders, x and y
    """
    x = tf.placeholder(tf.float32, shape=(None, nx), name="x")
    y = tf.placeholder(tf.float32, shape=(None, classes), name="y")

    return x, y
