#!/usr/bin/env python3
"""
module containing function forward_prop, shuffle_data and model
"""
import numpy as np
import tensorflow.compat.v1 as tf


def create_Adam_op(loss, global_step, alpha, beta1, beta2, epsilon):
    """
    function that creates the training operation for a neural network
        in tensorflow using the Adam optimization algorithm
    Args:
        loss: the loss of the network
        alpha: the learning rate
        beta1: the weight used for the first moment
        beta2: the weight used for the second moment
        epsilon: a small number to avoid division by zero
    Return: the Adam optimization operation
    """
    optimizer = tf.train.AdamOptimizer(learning_rate=alpha,
                                       beta1=beta1,
                                       beta2=beta2,
                                       epsilon=epsilon,
                                       name='Adam')
    adam_op = optimizer.minimize(loss, global_step=global_step)
    return adam_op


def learning_rate_decay(alpha, decay_rate, global_step, decay_step):
    """
    function that creates a learning rate decay operation
        in tensorflow using inverse time decay
    Args:
        alpha: the original learning rate
        decay_rate: the weight used to determine the rate
            at which alpha will decay
        global_step: the number of passes of gradient descent that have elapsed
        decay_step: the number of passes of gradient descent
            that should occur before alpha is decayed further
    Return: the learning rate decay operation
    """
    learning_rate_decay_op = tf.train.inverse_time_decay(
        learning_rate=alpha,
        global_step=global_step,
        decay_steps=decay_step,
        decay_rate=decay_rate,
        staircase=True)

    return learning_rate_decay_op


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


def calculate_loss(y, y_pred):
    """
    function that calculates the softmax cross-entropy loss of a prediction
    Args:
        y: a placeholder for the labels of the input data
        y_pred: a tensor containing the network's predictions
    Return: a tensor containing the loss of the prediction
    """
    return tf.losses.softmax_cross_entropy(y, y_pred)


def create_layer(prev, n):
    """
    function that create a layer in the neural network
    Args:
        prev: the tensor output of the previous layer
        n: the number of nodes in the layer to create
        activation: the activation function that the layer should use
    Return: the tensor output of the layer
    """
    layer_weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.layers.Dense(
        n,
        activation=None,
        kernel_initializer=layer_weights,
        name="layer"
    )

    return layer(prev)


def create_batch_norm_layer(prev, n, activation, epsilon):
    """
    function that creates a batch normalization layer
        for a neural network in tensorflow
    Args:
        prev: the activated output of the previous layer
        n: the number of nodes in the layer to be created
        activation: the activation function that should be
            used on the output of the layer
    Return: a tensor of the activated output for the layer
    """
    layer_weights = tf.keras.initializers.VarianceScaling(mode='fan_avg')

    layer = tf.layers.Dense(
        n,
        activation=None,
        kernel_initializer=layer_weights,
        name="layer")

    batch_mean, batch_variance = tf.nn.moments(layer(prev), [0])
    beta = tf.Variable(tf.zeros(shape=[n]))
    gamma = tf.Variable(tf.ones(shape=[n]))

    batch_norm = tf.nn.batch_normalization(
        x=layer(prev),
        mean=batch_mean,
        variance=batch_variance,
        offset=beta,
        scale=gamma,
        variance_epsilon=epsilon)

    return activation(batch_norm)


def forward_prop(prev, layers, activations, epsilon):
    """
    function that creates the forward propagation graph
        for the neural network
    Args:
        prev: the activated output of the previous layer
        layers: list containing the number of nodes in each layer
            of the network
        activations: list containing the activation functions
            for each layer of the network
        epsilon: a small number used to avoid division by zero
    Return: the prediction of the network in tensor form
    """
    # all layers get batch_normalization but the last one,
    #   that stays without any activation or normalization
    for i in range(len(layers) - 1):
        prev = create_batch_norm_layer(prev,
                                       layers[i],
                                       activations[i],
                                       epsilon)

    prev = create_layer(prev, layers[len(layers) - 1])

    return prev


def shuffle_data(X, Y):
    """
    function that shuffles the data points in two matrices the same way
    Args:
        X: the first numpy.ndarray of shape (m, nx) to shuffle
            m: the number of data points
            nx: the number of features in X
        Y: the second numpy.ndarray of shape (m, ny) to shuffle
            m: the same number of data points as in X
            ny: the number of features in Y
    Return: the shuffled X and Y matrices
    """
    permutation = np.random.permutation(X.shape[0])
    shuffled_X = X[permutation]
    shuffled_Y = Y[permutation]

    return shuffled_X, shuffled_Y


def model(Data_train, Data_valid, layers, activations, alpha=0.001, beta1=0.9,
          beta2=0.999, epsilon=1e-8, decay_rate=1, batch_size=32, epochs=5,
          save_path='/tmp/model.ckpt'):
    """
    function that  builds, trains, and saves a neural network model
        in tensorflow using Adam optimization, mini-batch gradient descent,
        learning rate decay, and batch normalization
    Args:
        Data_train: tuple containing the training inputs and training labels
        Data_valid: tuple containing the validation inputs
            and validation labels
        layers: list containing the number of nodes in each layer
            of the network
        activation: list containing the activation functions
            used for each layer of the network
        alpha: learning rate
        beta1: weight for the first moment of Adam Optimization
        beta2: weight for the second moment of Adam Optimization
        epsilon: small number used to avoid division by zero
        decay_rate: decay rate for inverse time decay of the learning rat
            (the corresponding decay step should be 1)
        batch_size: number of data points that should be in a mini-batch
        epochs: number of times the training should pass through
            the whole dataset
        save_path: path where the model should be saved to
    Return: the path where the model was saved
    """
    # get X_train, Y_train, X_valid, and Y_valid from Data_train and Data_valid
    X_train = Data_train[0]
    Y_train = Data_train[1]
    X_valid = Data_valid[0]
    Y_valid = Data_valid[1]

    # initialize x, y and add them to collection
    x = tf.placeholder(tf.float32, shape=(None, X_train.shape[1]), name="x")
    y = tf.placeholder(tf.float32, shape=(None, Y_train.shape[1]), name="y")
    tf.add_to_collection('x', x)
    tf.add_to_collection('y', y)

    # initialize y_pred and add it to collection
    y_pred = forward_prop(x, layers, activations, epsilon)
    tf.add_to_collection('y_pred', y_pred)

    # intialize loss and add it to collection
    loss = calculate_loss(y, y_pred)
    tf.add_to_collection('loss', loss)

    # intialize accuracy and add it to collection
    accuracy = calculate_accuracy(y, y_pred)
    tf.add_to_collection('accuracy', accuracy)

    # intialize global_step variable
    # hint: not trainable
    global_step = tf.Variable(0, trainable=False)

    # compute decay_steps
    if X_train.shape[0] % batch_size == 0:
        decay_step = X_train.shape[0] // batch_size
    else:
        decay_step = (X_train.shape[0] // batch_size) + 1

    # create "alpha" the learning rate decay operation in tensorflow
    alpha = learning_rate_decay(alpha, decay_rate, global_step, decay_step)

    # initizalize train_op and add it to collection
    # hint: don't forget to add global_step parameter in optimizer().minimize()
    train_op = create_Adam_op(loss, global_step, alpha, beta1, beta2, epsilon)
    tf.add_to_collection('train_op', train_op)

    init = tf.global_variables_initializer()

    with tf.Session() as sess:
        sess.run(init)

        for i in range(epochs):
            # print training and validation cost and accuracy
            training_cost = sess.run(
                loss,
                feed_dict={x: X_train, y: Y_train})
            training_accuracy = sess.run(
                accuracy,
                feed_dict={x: X_train, y: Y_train})
            validation_cost = sess.run(
                loss,
                feed_dict={x: X_valid, y: Y_valid})
            validation_accuracy = sess.run(
                accuracy,
                feed_dict={x: X_valid, y: Y_valid})

            print("After {} epochs:".format(i))
            print("\tTraining Cost: {}".format(training_cost))
            print("\tTraining Accuracy: {}".format(training_accuracy))
            print("\tValidation Cost: {}".format(validation_cost))
            print("\tValidation Accuracy: {}".format(validation_accuracy))

            # shuffle data
            shuffled_X, shuffled_Y = shuffle_data(X_train, Y_train)

            step = 0
            for j in range(0, X_train.shape[0], batch_size):
                # get X_batch and Y_batch
                #   from X_train shuffled and Y_train shuffled
                X_batch = shuffled_X[j:j + batch_size]
                Y_batch = shuffled_Y[j:j + batch_size]

                # run training operation
                sess.run(train_op, feed_dict={x: X_batch, y: Y_batch})

                # print batch cost and accuracy
                if step > 0 and step % 100 == 0:
                    step_cost = sess.run(
                        loss, feed_dict={x: X_batch, y: Y_batch})
                    step_accuracy = sess.run(
                        accuracy, feed_dict={x: X_batch, y: Y_batch})
                    print("\tStep {}:".format(step))
                    print("\t\tCost: {}".format(step_cost))
                    print("\t\tAccuracy: {}".format(step_accuracy))

                step += 1

        # print training and validation cost and accuracy again
        training_cost = sess.run(
            loss,
            feed_dict={x: X_train, y: Y_train})
        training_accuracy = sess.run(
            accuracy,
            feed_dict={x: X_train, y: Y_train})
        validation_cost = sess.run(
            loss,
            feed_dict={x: X_valid, y: Y_valid})
        validation_accuracy = sess.run(
            accuracy,
            feed_dict={x: X_valid, y: Y_valid})

        print("After {} epochs:".format(i + 1))
        print("\tTraining Cost: {}".format(training_cost))
        print("\tTraining Accuracy: {}".format(training_accuracy))
        print("\tValidation Cost: {}".format(validation_cost))
        print("\tValidation Accuracy: {}".format(validation_accuracy))

        # save and return the path to where the model was saved
        saver = tf.train.Saver()
        return saver.save(sess, save_path)
