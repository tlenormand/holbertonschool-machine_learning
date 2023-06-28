#!/usr/bin/env python3
""" Multivariate Probability """

import numpy as np

mean_cov = __import__('0-mean_cov').mean_cov


class MultiNormal():
    """ represents a Multivariate Normal distribution """
    mean = None
    cov = None

    def __init__(self, data):
        """ constructor

        Argumnets:
            data (numpy.ndarray): array of shape (d, n) containing the data set
                n is the number of data points
                d is the number of dimensions in each data point
        """
        dataT = data.T

        if type(dataT) is not np.ndarray or len(dataT.shape) != 2:
            raise TypeError('data must be a 2D numpy.ndarray')

        n, d = dataT.shape
        if n < 2:
            raise ValueError('data must contain multiple data points')

        mean, cov = mean_cov(dataT)
        self.mean = mean.T
        self.cov = cov

    def pdf(self, x):
        """ calculates the PDF at a data point

        Arguments:
            x (numpy.ndarray): array of shape (d, 1) containing the data point
                whose PDF should be calculated
                d is the number of dimensions of the Multinomial instance

        Returns:
            pdf (float): the value of the PDF
        """
        if type(x) is not np.ndarray:
            raise TypeError('x must be a numpy.ndarray')

        d = self.cov.shape[0]
        if len(x.shape) != 2 or x.shape != (d, 1):
            raise ValueError('x must have the shape ({}, 1)'.format(d))

        # pdf = 1 / sqrt((2pi)^d * det(cov)) *
        # e^(-1/2 * (x - mean).T * cov^-1 * (x - mean))
        pdf = (1 / np.sqrt(((2 * np.pi) ** d) * np.linalg.det(self.cov)) *
               np.exp(-(np.linalg.solve(self.cov, x - self.mean).
                        T.dot(x - self.mean)) / 2))

        return pdf[0][0]
