#!/usr/bin/env python3
""" Module that defines a class called Neuron """

import numpy as np


class Neuron:
    """ Class Neuron

    Private instance attributes:
        __W: The weights vector for the neuron
        __b: The bias for the neuron
        __A: The activated output of the neuron (prediction)

    Methods:
        __init__: Constructor
    """
    def __init__(self, nx):
        """ Constructor

        Arguments:
            nx: number of input features to the neuron
        """
        self.__W = 0
        self.__b = 0
        self.__A = 0
        self.nx = nx

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.__W = np.random.normal(loc=0, scale=1, size=(1, nx))

###############################################################################
# Getters and Setters
###############################################################################

    @property
    def A(self):
        """ Getter for A """
        return self.__A

    @A.setter
    def A(self, value):
        """ Setter for A """
        self.__A = value

    @property
    def b(self):
        """ Getter for b """
        return self.__b

    @b.setter
    def b(self, value):
        """ Setter for b """
        self.__b = value

    @property
    def W(self):
        """ Getter for W """
        return self.__W

    @W.setter
    def W(self, value):
        """ Setter for W """
        self.__W = value

###############################################################################
# Public Methods
###############################################################################

    def forward_prop(self, X):
        """ Calculates the forward propagation of the neuron

        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input data

        Returns:
            The private attribute __A
        """
        Z = np.matmul(self.__W, X) + self.__b
        self.__A = 1 / (1 + np.exp(-Z))
        return self.__A

    def cost(self, Y, A):
        """ Calculates the cost of the model using logistic regression

        Arguments:
            Y: numpy.ndarray with shape (1, m) that contains the correct
                labels for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
                of the neuron for each example

        Returns:
            The cost
        """
        m = Y.shape[1]
        return (1 / m * np.sum(
            -Y * np.log(A) - (1 - Y) * np.log(1.0000001 - A)
        ))

    def evaluate(self, X, Y):
        """ Evaluates the neuron's predictions

        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input data
            Y: numpy.ndarray with shape (1, m) that contains the correct
                labels for the input data

        Returns:
            The neuron's prediction and the cost of the network
        """
        A = self.forward_prop(X)
        return np.where(A >= 0.5, 1, 0), self.cost(Y, A)

    def gradient_descent(self, X, Y, A, alpha=0.05):
        """ Calculates one pass of gradient descent on the neuron

        Arguments:
            X: numpy.ndarray with shape (nx, m) that contains the input data
            Y: numpy.ndarray with shape (1, m) that contains the correct
                labels for the input data
            A: numpy.ndarray with shape (1, m) containing the activated output
                of the neuron for each example
            alpha: the learning rate
        """
        m = Y.shape[1]
        dW = 1 / m * np.matmul((A - Y), X.T)
        db = 1 / m * np.sum(A - Y)
        self.__W = self.__W - alpha * dW
        self.__b = self.__b - alpha * db
