#!/usr/bin/env python3
""" Module that defines a class called DeepNeuralNetwork """

import numpy as np
import matplotlib.pyplot as plt
import pickle


class DeepNeuralNetwork:
    """ Class DeepNeuralNetwork

    Methods:
        __init__: Constructor
    """
    def __init__(self, nx, layers):
        """ Constructor

        Arguments:
            nx: number of input features to the neuron
            layers: list representing the number of nodes in each layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        if not isinstance(layers, list) or len(layers) == 0:
            raise TypeError("layers must be a list of positive integers")

        self.__L = len(layers)
        self.__cache = {}
        self.__weights = {}

        for i in range(len(layers)):
            if not isinstance(layers[i], int) or layers[i] <= 0:
                raise TypeError("layers must be a list of positive integers")

            if i == 0:
                self.__weights["W{}".format(i + 1)] = \
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.__weights["W{}".format(i + 1)] = \
                    (np.random.randn(layers[i], layers[i - 1]) *
                     np.sqrt(2 / layers[i - 1]))

            self.__weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))

##############################################################################
# Getters and Setters
##############################################################################

    @property
    def L(self):
        """ Getter for L """
        return self.__L

    @property
    def cache(self):
        """ Getter for cache """
        return self.__cache

    @property
    def weights(self):
        """ Getter for weights """
        return self.__weights

##############################################################################
# Public instance methods
##############################################################################

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network

        Arguments:
            X: input data

        Returns:
            The output of the neural network and the cache, respectively
        """
        self.__cache["A0"] = X

        for i in range(self.__L):
            Zx = np.matmul(
                    self.__weights["W{}".format(i + 1)],
                    self.__cache["A{}".format(i)]
                ) + self.__weights["b{}".format(i + 1)]

            if i == self.__L - 1:
                # softmax activation for last layer
                t = np.exp(Zx)
                self.__cache["A{}".format(i + 1)] = \
                    t / np.sum(t, axis=0, keepdims=True)
            else:
                # ReLU activation for hidden layers
                self.__cache["A{}".format(i + 1)] = np.maximum(0, Zx)

        return self.__cache["A{}".format(self.__L)], self.__cache

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression

        Arguments:
            Y: correct labels for the input data
            A: activated output of the neuron for each example

        Returns:
            The cost
        """
        m = Y.shape[1]

        return -1 / m * np.sum(Y * np.log(A))

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions

        Arguments:
            X: input data
            Y: correct labels for the input data

        Returns:
            The neuron's prediction and the cost of the network, respectively
        """
        A_last, cache = self.forward_prop(X)

        max = np.amax(A_last, axis=0, keepdims=True)

        return np.where(A_last >= 0.5, 1, 0), self.cost(max, A_last)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network

        Arguments:
            Y: correct labels for the input data
            cache: dictionary containing all the intermediary values of the
                network
            alpha: learning rate
        """
        W_tmp = None
        m = Y.shape[1]

        # first iteration
        A = cache["A{}".format(self.__L)]
        # dZ = A - Y
        dZ = A - Y

        for i in reversed(range(1, self.__L + 1)):  # i: 3, 2, 1 / L : 3

            if W_tmp is not None:
                A = cache["A{}".format(i)]
                # dZ = np.matmul(Wx.T, dZ) * (A * (1 - A))
                dZ = np.matmul(W_tmp.T, dZ) * (A * (1 - A))

            # dW = np.matmul(dZ, A.T) / m
            dW = np.matmul(dZ, cache["A{}".format(i - 1)].T) / m

            # db = np.sum(dZ) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # keep in tmp current weight
            W_tmp = self.__weights.get("W{}".format(i))

            # W = W - alpha * dW
            self.__weights["W{}".format(i)] = \
                self.__weights["W{}".format(i)] - alpha * dW

            # b = b - alpha * db
            self.__weights["b{}".format(i)] = \
                self.__weights["b{}".format(i)] - alpha * db

    def train(self, X, Y, iterations=5000, alpha=0.05,
              verbose=True, graph=True, step=100):
        """ Trains the deep neural network

        Arguments:
            X: input data
            Y: correct labels for the input data
            iterations: number of iterations to train over
            alpha: learning rate
            verbose: boolean that defines whether or not to print information
                about the training
            graph: boolean that defines whether or not to graph information
                about the training once the training has completed
            step: number of iterations between printing and graphing
                information

        Returns:
            The evaluation of the training data after iterations of training
                have occurred
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations <= 0:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if alpha < 0:
            raise ValueError("alpha must be positive")

        if not isinstance(step, int):
            raise TypeError("step must be an integer")

        if step < 1 or step > iterations:
            step = iterations

        costs = []
        for i in range(iterations + 1):
            self.forward_prop(X)

            if i % step == 0 or i == iterations:
                costs.append(
                    self.cost(Y, self.__cache["A{}".format(self.__L)])
                )

                if verbose:
                    print("Cost after {} iterations: {}".format(i, costs[-1]))

                if graph:
                    plt.plot(np.arange(0, i + 1, step), costs)
                    plt.xlabel("iteration")
                    plt.ylabel("cost")
                    plt.title("Training Cost")

            # don't execute last gradient_descent
            if i == iterations:
                return self.evaluate(X, Y)

            self.gradient_descent(Y, self.__cache, alpha)

        plt.show()

        return self.evaluate(X, Y)

    def save(self, filename):
        """ Saves the instance object to a file in pickle format

        Arguments:
            filename: file to which the object should be saved
        """
        if not filename.endswith(".pkl"):
            filename += ".pkl"

        with open(filename, "wb") as f:  # wb: write binary
            pickle.dump(self, f)

    def load(filename):
        """ Loads a pickled DeepNeuralNetwork object

        Arguments:
            filename: file from which the object should be loaded

        Returns:
            The loaded object
        """
        try:
            with open(filename, "rb") as f:  # rb: read binary
                return pickle.load(f)

        except FileNotFoundError:
            return None
