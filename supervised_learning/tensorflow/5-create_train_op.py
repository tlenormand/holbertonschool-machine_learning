#!/usr/bin/env python3
"""
module containing function create_train_op
"""
import tensorflow as tf


def create_train_op(loss, alpha):
    """
    function that creates the training operation for the network
    Args:
        loss: the loss of the network's prediction
        alpha: the learning rate
    Return: an operation that trains the network using gradient descent
    """
    optimizer = tf.train.GradientDescentOptimizer(learning_rate=alpha,
                                                  name='GradientDescent')
    train = optimizer.minimize(loss)
    return train
