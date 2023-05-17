#!/usr/bin/env python3
""" LeNet-5 (Tensorflow) """

import tensorflow.compat.v1 as tf
tf.disable_v2_behavior()

def lenet5(x, y):
    """ function that builds a modified version of the LeNet-5 architecture
        using tensorflow

    Arguments:
        x: tf.placeholder of shape (m, 28, 28, 1) containing the input images
            for the network
            m: the number of images
        y: tf.placeholder of shape (m, 10) containing the one-hot labels for
            the network

    Returns:
        a tensor for the softmax activated output
        a training operation that utilizes Adam optimization (with default
            hyperparameters)
        a tensor for the loss of the netowrk
        a tensor for the accuracy of the network
    """
    # initializer = tf.contrib.layers.variance_scaling_initializer()
    # initializer = tf.keras.initializers.VarianceScaling(scale=2.0)
    initializer = tf.truncated_normal_initializer(stddev=0.1)

    # Convolutional layer 1
    conv1 = tf.layers.Conv2D(
        filters=6,
        kernel_size=(5, 5),
        padding='same',
        activation=tf.nn.relu,
        kernel_initializer=initializer,
    )(x)

    # Pooling layer 1
    pool1 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(conv1)

    # Convolutional layer 2
    conv2 = tf.layers.Conv2D(
        filters=16,
        kernel_size=(5, 5),
        padding='valid',
        activation=tf.nn.relu,
        kernel_initializer=initializer,
    )(pool1)

    # Pooling layer 2
    pool2 = tf.layers.MaxPooling2D(
        pool_size=(2, 2),
        strides=(2, 2),
    )(conv2)

    # Flatten
    flatten = tf.layers.Flatten()(pool2)

    # Fully connected layer 1
    fc1 = tf.layers.Dense(
        units=120,
        activation=tf.nn.relu,
        kernel_initializer=initializer,
    )(flatten)

    # Fully connected layer 2
    fc2 = tf.layers.Dense(
        units=84,
        activation=tf.nn.relu,
        kernel_initializer=initializer,
    )(fc1)

    # Fully connected layer 3
    fc3 = tf.layers.Dense(
        units=10,
        kernel_initializer=initializer,
    )(fc2)

    # Softmax
    softmax = tf.nn.softmax(fc3)

    # Loss
    loss = tf.losses.softmax_cross_entropy(y, fc3)

    # Adam optimizer
    train_op = tf.train.AdamOptimizer().minimize(loss)

    # Accuracy
    y_pred = tf.argmax(softmax, 1)
    y_true = tf.argmax(y, 1)
    equality = tf.equal(y_pred, y_true)
    accuracy = tf.reduce_mean(tf.cast(equality, tf.float32))

    return softmax, train_op, loss, accuracy
