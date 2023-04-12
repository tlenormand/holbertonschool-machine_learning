#!/usr/bin/env python3
""" Module that defines a class called DeepNeuralNetwork """

import numpy as np


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

            self.__cache["A{}".format(i + 1)] = 1 / (1 + np.exp(-Zx))

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

        return (1 / m * np.sum(
            -Y * np.log(A) - (1 - Y) * np.log(1.0000001 - A)
        ))

    def evaluate(self, X, Y):
        """ Evaluates the neural network's predictions

        Arguments:
            X: input data
            Y: correct labels for the input data

        Returns:
            The neuron's prediction and the cost of the network, respectively
        """
        A_last, cache = self.forward_prop(X)

        return np.where(A_last >= 0.5, 1, 0), self.cost(Y, A_last)

    def gradient_descent(self, Y, cache, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network

        Arguments:
            Y: correct labels for the input data
            cache: dictionary containing all the intermediary values of the
                network
            alpha: learning rate
        """
        m = Y.shape[1]

        for i in reversed(range(self.__L)): # i: 2, 1, 0 / L : 3
            W_tmp = self.__weights.get("W{}".format(i + 2))

            # first iteration
            if i == self.__L - 1:
                A = cache["A{}".format(self.__L)]
                # dZ = A - Y
                dZ = A - Y
            else:
                A = cache["A{}".format(i + 1)]
                # dZ = np.matmul(Wx.T, dZ) * (A * (1 - A))
                dZ = np.matmul(
                    W_tmp.T, dZ
                ) * (A * (1 - A))

            # dW = np.matmul(dZ, A.T) / m
            dW = np.matmul(dZ, cache["A{}".format(i)].T) / m

            # db = np.sum(dZ) / m
            db = np.sum(dZ, axis=1, keepdims=True) / m

            # W = W - alpha * dW
            self.__weights["W{}".format(i + 1)] -= alpha * dW

            # b = b - alpha * db
            self.__weights["b{}".format(i + 1)] -= alpha * db
