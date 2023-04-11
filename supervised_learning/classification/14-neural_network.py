#!/usr/bin/env python3
""" Module that defines a class called NeuralNetwork """

import numpy as np


class NeuralNetwork:
    """ Class NeuralNetwork

    Methods:
        __init__: Constructor
    """
    def __init__(self, nx, nodes):
        """ Constructor

        Arguments:
            nx: number of input features to the neuron
            nodes: number of nodes found in the hidden layer
        """
        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx <= 0:
            raise ValueError("nx must be a positive integer")

        if not isinstance(nodes, int):
            raise TypeError("nodes must be an integer")

        if nodes < 1:
            raise ValueError("nodes must be a positive integer")

        self.__W1 = np.random.normal(size=(nodes, nx))
        self.__b1 = np.zeros((nodes, 1))
        self.__A1 = 0
        self.__W2 = np.random.normal(size=(1, nodes))
        self.__b2 = 0
        self.__A2 = 0

##############################################################################
# Getters and Setters
##############################################################################

    @property
    def W1(self):
        """ Getter for W1 """
        return self.__W1

    @property
    def b1(self):
        """ Getter for b1 """
        return self.__b1

    @property
    def A1(self):
        """ Getter for A1 """
        return self.__A1

    @property
    def W2(self):
        """ Getter for W2 """
        return self.__W2

    @property
    def b2(self):
        """ Getter for b2 """
        return self.__b2

    @property
    def A2(self):
        """ Getter for A2 """
        return self.__A2

##############################################################################
# Public instance methods
##############################################################################

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neural network

        Arguments:
            X: input data

        Returns:
            The private attributes __A1 and __A2
        """
        Z1 = np.matmul(self.__W1, X) + self.__b1
        self.__A1 = 1 / (1 + np.exp(-Z1))

        Z2 = np.matmul(self.__W2, self.__A1) + self.__b2
        self.__A2 = 1 / (1 + np.exp(-Z2))

        return self.__A1, self.__A2

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
            The neuron's prediction and the cost of the network
        """
        A1, A2 = self.forward_prop(X)
        return np.where(A2 >= 0.5, 1, 0), self.cost(Y, A2)

    def gradient_descent(self, X, Y, A1, A2, alpha=0.05):
        """ Calculates one pass of gradient descent on the neural network

        Arguments:
            X: input data
            Y: correct labels for the input data
            A1: output of the hidden layer
            A2: predicted output
            alpha: learning rate
        """
        m = Y.shape[1]

        dZ2 = A2 - Y
        dW2 = 1 / m * np.matmul(dZ2, A1.T)
        db2 = 1 / m * np.sum(dZ2, axis=1, keepdims=True)

        dZ1 = np.matmul(self.__W2.T, dZ2) * (A1 * (1 - A1))
        dW1 = 1 / m * np.matmul(dZ1, X.T)
        db1 = 1 / m * np.sum(dZ1, axis=1, keepdims=True)

        self.__W2 = self.__W2 - alpha * dW2
        self.__b2 = self.__b2 - alpha * db2
        self.__W1 = self.__W1 - alpha * dW1
        self.__b1 = self.__b1 - alpha * db1

    def train(self, X, Y, iterations=5000, alpha=0.05):
        """ Trains the neural network

        Arguments:
            X: input data
            Y: correct labels for the input data
            iterations: number of iterations to train over
            alpha: learning rate

        Returns:
            The evaluation of the training data after iterations of training
            have occurred
        """
        if not isinstance(iterations, int):
            raise TypeError("iterations must be an integer")

        if iterations < 1:
            raise ValueError("iterations must be a positive integer")

        if not isinstance(alpha, float):
            raise TypeError("alpha must be a float")

        if alpha <= 0:
            raise ValueError("alpha must be positive")

        for i in range(iterations):
            self.forward_prop(X)
            self.gradient_descent(X, Y, self.__A1, self.__A2, alpha)

        return self.evaluate(X, Y)
