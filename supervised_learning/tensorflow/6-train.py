#!/usr/bin/env python3
"""
module containing function train
"""
import tensorflow as tf

create_placeholders = __import__('0-create_placeholders').create_placeholders
forward_prop = __import__('2-forward_prop').forward_prop
calculate_accuracy = __import__('3-calculate_accuracy').calculate_accuracy
calculate_loss = __import__('4-calculate_loss').calculate_loss
create_train_op = __import__('5-create_train_op').create_train_op


def train(X_train, Y_train, X_valid, Y_valid, layer_sizes, activations,
          alpha, iterations, save_path="/tmp/model.ckpt"):
    """
    function that builds, trains, and saves a neural network classifier
    Args:
        X_train: numpy.ndarray containing the training input data
        Y_train: numpy.ndarray containing the training labels
        X_valid: numpy.ndarray containing the validation input data
        Y_valid: numpy.ndarray containing the validation labels
        layer_sizes: list containing the number of nodes in each layer
        activations: list containing the activation functions for each layer
        alpha: the learning rate
        iterations: the number of iterations to train over
        save_path: path to save the model
    Return: the path where the model was saved
    """
    x, y = create_placeholders(X_train.shape[1], Y_train.shape[1])
    y_pred = forward_prop(x, layer_sizes, activations)
    loss = calculate_loss(y, y_pred)
    accuracy = calculate_accuracy(y, y_pred)
    train_op = create_train_op(loss, alpha)

    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)
    tf.add_to_collection('y_pred', y_pred)
    tf.add_to_collection('loss', loss)
    tf.add_to_collection('accuracy', accuracy)
    tf.add_to_collection('train_op', train_op)

    # Add an op to initialize the variables.
    init_op = tf.global_variables_initializer()

    # Add ops to save and restore all the variables.
    saver = tf.train.Saver()

    with tf.Session() as session:
        session.run(init_op)

        for i in range(iterations + 1):
            training_cost = session.run(
                loss,
                feed_dict={x: X_train, y: Y_train})
            training_accuracy = session.run(
                accuracy,
                feed_dict={x: X_train, y: Y_train})
            validation_cost = session.run(
                loss,
                feed_dict={x: X_valid, y: Y_valid})
            validation_accuracy = session.run(
                accuracy,
                feed_dict={x: X_valid, y: Y_valid})

            if i % 100 == 0 or i == iterations:
                print("After {} iterations:".format(i))
                print("\tTraining Cost: {}".format(training_cost))
                print("\tTraining Accuracy: {}".format(training_accuracy))
                print("\tValidation Cost: {}".format(validation_cost))
                print("\tValidation Accuracy: {}".format(validation_accuracy))

            if i < iterations:
                session.run(train_op, feed_dict={x: X_train, y: Y_train})

        save_path = saver.save(session, save_path)
    return save_path
