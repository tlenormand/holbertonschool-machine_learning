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

        if all(layer <= 0 for layer in layers):
            raise TypeError("layers must be a list of positive integers")

        self.L = len(layers)
        self.cache = {}
        self.weights = {}

        for i in range(len(layers)):
            if i == 0:
                self.weights["W{}".format(i + 1)] = \
                    np.random.randn(layers[i], nx) * np.sqrt(2 / nx)
            else:
                self.weights["W{}".format(i + 1)] = \
                    (np.random.randn(layers[i], layers[i - 1]) *
                     np.sqrt(2 / layers[i - 1]))

            self.weights["b{}".format(i + 1)] = np.zeros((layers[i], 1))
