#!/usr/bin/env python3
"""
module containing function calculate_accuracy
"""
import tensorflow.compat.v1 as tf


def calculate_accuracy(y, y_pred):
    """
    function that calculates the accuracy of a prediction
    Args:
        y: a placeholder for the labels of the input data
        y_pred: a tensor containing the network's predictions
    Return: a tensor containing the decimal accuracy of the prediction
    """
    y_tensor = tf.argmax(y, axis=1)
    y_pred_tensor = tf.argmax(y_pred, axis=1)

    correct_predictions = tf.math.equal(y_tensor, y_pred_tensor)
    accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

    return accuracy
