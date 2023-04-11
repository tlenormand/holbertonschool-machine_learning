#!/usr/bin/env python3
""" Module that defines a class called Neuron """

import numpy as np


class Neuron:
    """ Class Neuron

    Public instance attributes:
        W: The weights vector for the neuron
        b: The bias for the neuron
        A: The activated output of the neuron (prediction)

    Methods:
        __init__: Constructor
    """
    def __init__(self, nx):
        """ Constructor

        Arguments:
            nx: number of input features to the neuron
        """
        self.W = 0
        self.b = 0
        self.A = 0

        if not isinstance(nx, int):
            raise TypeError("nx must be an integer")

        if nx < 1:
            raise ValueError("nx must be a positive integer")

        self.W = np.random.normal(loc=0, scale=1, size=(1, nx))
