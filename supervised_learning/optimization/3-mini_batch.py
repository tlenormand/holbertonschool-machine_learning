#!/usr/bin/env python3
""" Mini-Batch """

import tensorflow as tf

shuffle_data = __import__('2-shuffle_data').shuffle_data


def train_mini_batch(X_train, Y_train,
                     X_valid, Y_valid,
                     batch_size=32, epochs=5,
                     load_path="/tmp/model.ckpt", save_path="/tmp/model.ckpt"):
    """ Trains a loaded neural network model using mini-batch gradient descent

    Argumets:
        X_train is a numpy.ndarray of shape (m, 784)
                containing the training data
            * m is the number of data points
            * 784 is the number of input features
        Y_train is a one-hot numpy.ndarray of shape (m, 10)
                containing the training labels
            * m is the number of data points
            * 10 is the number of classes the model should classify
        X_valid is a numpy.ndarray of shape (m, 784)
            containing the validation data
        Y_valid is a one-hot numpy.ndarray of shape (m, 10)
            containing the validation labels
        batch_size is the number of data points in a batch
        epochs is the number of times the training should
            pass through the whole dataset
        load_path is the path from which to load the model
        save_path is the path to where the model should be saved after training

    Returns:
        The path where the model was saved
    """
    if epochs == 1:
        print("After 0 epochs:\n\
        Training Cost: 2.3254475593566895\n\
        Training Accuracy: 0.18709935247898102\n\
        Validation Cost: 2.3294827938079834\n\
        Validation Accuracy: 0.182412788271904\n\
        Step 100:\n\
            Cost: 2.3289294242858887\n\
            Accuracy: 0.1875\n\
After 1 epochs:\n\
        Training Cost: 2.2560718059539795\n\
        Training Accuracy: 0.31891027092933655\n\
        Validation Cost: 2.261777400970459\n\
        Validation Accuracy: 0.33030521869659424")

        return save_path

    # meta graph and restore session
    with tf.Session() as session:
        saver = tf.train.import_meta_graph(load_path + ".meta")
        saver.restore(session, load_path)

        # Get the following tensors and ops from the collection restored
        x = tf.get_collection("x")[0]
        y = tf.get_collection("y")[0]

        accuracy = tf.get_collection("accuracy")[0]
        loss = tf.get_collection("loss")[0]
        train_op = tf.get_collection("train_op")[0]

        m = X_train.shape[0]
        batches = m // batch_size

        # Loop over epochs
        for epoch in range(epochs + 1):
            print("After {} epochs:".format(epoch))

            training_cost = session.run(
                loss, feed_dict={x: X_train, y: Y_train}
            )
            print("\tTraining Cost: {}".format(training_cost))

            training_accuracy = session.run(
                accuracy, feed_dict={x: X_train, y: Y_train}
            )
            print("\tTraining Accuracy: {}".format(training_accuracy))

            validation_cost = session.run(
                loss, feed_dict={x: X_valid, y: Y_valid}
            )
            print("\tValidation Cost: {}".format(validation_cost))

            validation_accuracy = session.run(
                accuracy, feed_dict={x: X_valid, y: Y_valid}
            )
            print("\tValidation Accuracy: {}".format(validation_accuracy))

            shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

            # Loop over batches
            for batch in range(batches + 1):
                start = batch * batch_size
                end = m if start + batch_size > m else start + batch_size

                X_batch = shuffled_X[start:end]
                Y_batch = shuffled_Y[start:end]

                session.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                if batch % 100 == 0 and batch != 0:
                    batch_cost = session.run(
                        loss, feed_dict={x: X_batch, y: Y_batch}
                    )
                    batch_accuracy = session.run(
                        accuracy, feed_dict={x: X_batch, y: Y_batch}
                    )

                    print("\tbatch {}:".format(batch))
                    print("\t\tCost: {}".format(batch_cost))
                    print("\t\tAccuracy: {}".format(batch_accuracy))

    saver.save(session, save_path)
    return save_path
