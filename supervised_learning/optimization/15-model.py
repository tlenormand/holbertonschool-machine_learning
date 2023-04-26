#!/usr/bin/env python3
""" 0x03. Optimization """

import tensorflow as tf
import numpy as np


def create_layer(prev, n, activation):
    """ Function that creates a layer

    Args:
        prev (tensor): tensor output of the previous layer
        n (int): number of nodes in the layer to create
        activation (tensor): activation function that should be used on the
            output of the layer

    Returns:
        tensor: the tensor output of the layer
    """
    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    layer = tf.layers.Dense(
        units=n,
        activation=activation,
        name='layer',
        kernel_initializer=init
    )

    return layer(prev)


def forward_prop(x, layer, activations):
    """ Function that creates the forward propagation graph for the neural
        network

    Args:
        x (tensor): placeholder for the input data
        layer (list): list containing the number of nodes in each layer of the
            network
        activations (list): list containing the activation functions for each
            layer of the network

    Returns:
        tensor: prediction of the network in tensor form
    """
    # First layer
    y_pred = create_batch_norm_layer(x, layer[0], activations[0])

    for i in range(1, len(layer)):
        y_pred = create_batch_norm_layer(y_pred, layer[i], activations[i])

    return y_pred


def calculate_accuracy(y, y_pred):
    """ Function that calculates the accuracy of a prediction

    Args:
        y (tensor): placeholder for the labels of the input data
        y_pred (tensor): tensor containing the network's predictions

    Returns:
        tensor: tensor containing the decimal accuracy of the prediction
    """
    prediction = tf.equal(tf.argmax(y, axis=1), tf.argmax(y_pred, axis=1))
    accuracy = tf.reduce_mean(tf.cast(prediction, tf.float32))

    return accuracy


def calculate_loss(y, y_pred):
    """ Function that calculates the softmax cross-entropy loss of a
        prediction

    Args:
        y (tensor): placeholder for the labels of the input data
        y_pred (tensor): tensor containing the network's predictions

    Returns:
        tensor: tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def shuffle_data(X, Y):
    """ Function that shuffles the data points in two matrices the same way

    Args:
        X (tensor): first numpy.ndarray of shape (m, nx) to shuffle
            m is the number of data points
            nx is the number of features in X
        Y (tensor): second numpy.ndarray of shape (m, ny) to shuffle
            m is the same number of data points as in X
            ny is the number of features in Y

    Returns:
        tuple: the shuffled X and Y matrices
    """
    shuffler = np.random.permutation(len(X))

    return X[shuffler], Y[shuffler]


def create_Adam_op(loss, alpha, beta1, beta2, epsilon):
    """ Function that creates the training operation for a neural network in
        tensorflow using the Adam optimization algorithm

    Args:
        loss (tensor): loss of the network
        alpha (float): learning rate
        beta1 (float): weight used for the first moment
        beta2 (float): weight used for the second moment
        epsilon (float): small number to avoid division by zero

    Returns:
        tensor: Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(alpha, beta1, beta2, epsilon)
    adam = optimizer.minimize(loss)

    return adam


def create_batch_norm_layer(prev, n, activation):
    """ Function that creates a batch normalization layer for a neural network
        in tensorflow

    Args:
        prev (tensor): activated output of the previous layer
        n (int): number of nodes in the layer to be created
        activation (tensor): activation function that should be used on the
            output of the layer

    Returns:
        tensor: output of the layer
    """
    if activation is None:
        return create_layer(prev, n, activation)

    init = tf.contrib.layers.variance_scaling_initializer(mode="FAN_AVG")

    y = tf.layers.Dense(units=n, kernel_initializer=init, name='layer')
    x = y(prev)

    mean, variance = tf.nn.moments(x, axes=[0])
    gamma = tf.Variable(tf.constant(1.0, shape=[n]), trainable=True)
    beta = tf.Variable(tf.constant(0.0, shape=[n]), trainable=True)
    epsilon = 1e-8

    normalization = tf.nn.batch_normalization(
        x=x,
        mean=mean,
        variance=variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon,
        name='Z'
    )

    return activation(normalization)


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """ Function that creates a learning rate decay operation in tensorflow
        using inverse time decay

    Args:
        alpha (float): original learning rate
        decay_rate (float): weight used to determine the rate at which alpha
            will decay
        global_step (int): number of passes of gradient descent that have
            elapsed
        decay_step (int): number of passes of gradient descent that should
            occur before alpha is decayed further

    Returns:
        tensor: the learning rate decay operation
    """

    return tf.train.inverse_time_decay(
        alpha,
        global_step,
        decay_step,
        decay_rate,
        staircase=True
    )


def model(Data_train, Data_valid, layers, activations, alpha=0.001,
          beta1=0.9, beta2=0.999, epsilon=1e-8, decay_rate=1,
          batch_size=32, epochs=5, save_path='/tmp/model.ckpt'):
    """ Function that builds, trains, and saves a neural network model in
        tensorflow using Adam optimization, mini-batch gradient descent,
        learning rate decay, and batch normalization

    Args:
        Data_train (tuple): tuple containing the training inputs and labels
        Data_valid (tuple): tuple containing the validation inputs and labels
        layers (list): list containing the number of nodes in each layer of
            the network
        activations (list): list containing the activation functions used for
            each layer of the network
        alpha (float): learning rate
        beta1 (float): weight used for the first moment
        beta2 (float): weight used for the second moment
        epsilon (float): small number to avoid division by zero
        decay_rate (float): weight used to determine the rate at which alpha
            will decay
        batch_size (int): number of data points that should be in a mini-batch
        epochs (int): number of times the training should pass through the
            whole dataset
        save_path (str): path where the model should be saved to

    Returns:
        str: the path where the model was saved
    """
    X_train, Y_train = Data_train[0], Data_train[1]
    X_valid, Y_valid = Data_valid[0], Data_valid[1]

    steps = int(np.ceil(X_train.shape[0] / batch_size))

    x = tf.placeholder(tf.float32, shape=[None, X_train.shape[1]], name='x')
    tf.add_to_collection('x', x)

    y = tf.placeholder(tf.float32, shape=[None, Y_train.shape[1]], name='y')
    tf.add_to_collection('y', y)

    y_pred = forward_prop(x, layers, activations)
    tf.add_to_collection('y_pred', y_pred)

    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    global_step = tf.Variable(0, trainable=False)
    alpha = learning_rate_decay(alpha, decay_rate, global_step, 1)

    train_op = create_Adam_op(loss, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()
    # Add ops to save/restore variables.
    saver = tf.train.Saver()

    with tf.Session() as sess:
        sess.run(init)

        for epoch in range(epochs + 1):
            print("After {} epochs:".format(epoch))

            train_cost, train_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_train, y: Y_train}
            )
            print("\tTraining Cost: {}".format(train_cost))
            print("\tTraining Accuracy: {}".format(train_accuracy))

            valid_cost, valid_accuracy = sess.run(
                [loss, accuracy],
                feed_dict={x: X_valid, y: Y_valid}
            )
            print("\tValidation Cost: {}".format(valid_cost))
            print("\tValidation Accuracy: {}".format(valid_accuracy))

            if epoch < epochs:
                # learning rate decay
                sess.run(global_step.assign(epoch))

                # update learning rate
                sess.run(alpha)

                # shuffle data, both training set and labels
                X_shuffled, Y_shuffled = shuffle_data(X_train, Y_train)

                # mini-batch within epoch
                for step_number in range(steps):
                    # data selection mini batch from training set and labels
                    start = step_number * batch_size
                    end = (step_number + 1) * batch_size
                    if end > X_train.shape[0]:
                        end = X_train.shape[0]

                    X = X_shuffled[start:end]
                    Y = Y_shuffled[start:end]

                    # execute training for step
                    sess.run(train_op, feed_dict={x: X, y: Y})

                    if step_number != 0 and (step_number + 1) % 100 == 0:
                        # step_number is the number of times gradient
                        # descent has been run in the current epoch
                        print("\tStep {}:".format(step_number + 1))

                        # calculate cost and accuracy for step
                        step_cost, step_accuracy = sess.run(
                            [loss, accuracy],
                            feed_dict={x: X, y: Y}
                        )
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        return saver.save(sess, save_path)
