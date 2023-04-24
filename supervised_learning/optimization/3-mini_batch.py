#!/usr/bin/env python3
"""
module containing function train_mini_batch
"""
import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train, X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """
    function that trains a loaded neural network model
        using mini-batch gradient descent
    Args:
        X_train: numpy.ndarray of shape (m, 784)
            containing the training input data
            with m: the number of data points
                784: the number of input features
        Y_train: one-hot numpy.ndarray of shape (m, 10)
            containing the training labels
            with: 10 is the number of classes the model should classify
        X_valid: numpy.ndarray of shape (m, 784) containing the validation data
        Y_valid: one-hot numpy.ndarray of shape (m, 10)
            containing the validation labels
        batch_size: the number of data points in a batch
        epochs: the number of times the training should pass through
            the whole dataset
        load_path: the path from which to load the model
        save_path: the path to where the model should be saved after training
    Return: the path where the model was saved
    """
    # meta graph and restore session
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(session, load_path)

        # Get the following tensors and ops from the collection restored
        x = tf.get_collection('x')[0]
        y = tf.get_collection('y')[0]
        accuracy = tf.get_collection('accuracy')[0]
        loss = tf.get_collection('loss')[0]
        train_op = tf.get_collection('train_op')[0]

        # loop over epochs
        if X_train.shape[0] % batch_size == 0:
            iterations = X_train.shape[0] // batch_size
        else:
            iterations = (X_train.shape[0] // batch_size) + 1

        for epoch in range(epochs + 1):
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

            print("After {} epochs:".format(epoch))
            print("\tTraining Cost: {}".format(training_cost))
            print("\tTraining Accuracy: {}".format(training_accuracy))
            print("\tValidation Cost: {}".format(validation_cost))
            print("\tValidation Accuracy: {}".format(validation_accuracy))

            # shuffle data
            shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

            # loop over the batches
            if epoch < epochs:

                start = 0
                if batch_size < X_train.shape[0]:
                    end = batch_size
                else:
                    end = X_train.shape[0]

                for step_number in range(iterations):

                    X_batch = shuffled_X[start:end]
                    Y_batch = shuffled_Y[start:end]
                    session.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                    start += batch_size
                    if end + batch_size < X_train.shape[0]:
                        end += batch_size
                    else:
                        end = X_train.shape[0]

                    if step_number > 0 and step_number % 100 == 0:
                        step_cost = session.run(
                            loss, feed_dict={x: X_batch, y: Y_batch})
                        step_accuracy = session.run(
                            accuracy, feed_dict={x: X_batch, y: Y_batch})
                        print("\tStep {}:".format(step_number))
                        print("\t\tCost: {}".format(step_cost))
                        print("\t\tAccuracy: {}".format(step_accuracy))

        # save session
        save_path = saver.save(session, save_path)
        return save_path
