#!/usr/bin/env python3
"""
module containing function calculate_loss
"""
import tensorflow as tf


def calculate_loss(y, y_pred):
    """
    function that calculates the softmax cross-entropy loss of a prediction
    Args:
        y: a placeholder for the labels of the input data
        y_pred: a tensor containing the network's predictions
    Return: a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)
