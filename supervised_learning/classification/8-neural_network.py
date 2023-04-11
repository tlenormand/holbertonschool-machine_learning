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

        self.W1 = np.random.normal(size=(nodes, nx))
        self.b1 = 0
        self.A1 = 0
        self.W2 = np.random.normal(size=(1, nodes))
        self.b2 = 0
        self.A2 = 0
