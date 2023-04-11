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
        self.__b1 = 0
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

    @W1.setter
    def W1(self, value):
        """ Setter for W1 """
        raise AttributeError("can't set attribute")

    @property
    def b1(self):
        """ Getter for b1 """
        return self.__b1

    @b1.setter
    def b1(self, value):
        """ Setter for b1 """
        raise AttributeError("can't set attribute")

    @property
    def A1(self):
        """ Getter for A1 """
        return self.__A1

    @A1.setter
    def A1(self, value):
        """ Setter for A1 """
        raise AttributeError("can't set attribute")

    @property
    def W2(self):
        """ Getter for W2 """
        return self.__W2

    @W2.setter
    def W2(self, value):
        """ Setter for W2 """
        raise AttributeError("can't set attribute")

    @property
    def b2(self):
        """ Getter for b2 """
        return self.__b2

    @b2.setter
    def b2(self, value):
        """ Setter for b2 """
        raise AttributeError("can't set attribute")

    @property
    def A2(self):
        """ Getter for A2 """
        return self.__A2

    @A2.setter
    def A2(self, value):
        """ Setter for A2 """
        raise AttributeError("can't set attribute")

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
