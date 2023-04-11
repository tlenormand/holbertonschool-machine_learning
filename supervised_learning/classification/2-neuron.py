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
