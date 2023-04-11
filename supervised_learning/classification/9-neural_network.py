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
