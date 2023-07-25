#!/usr/bin/env python3
""" Initialize Gaussian Process """

import numpy as np


class GaussianProcess:
    """ represents a noiseless 1D Gaussian process """
    def __init__(self, X_init, Y_init, l=1, sigma_f=1):
        """ constructor

        Argumnets:
            X_init: np arr (t, 1) inputs sampled with black-box function
            Y_init: np arr (t, 1) outputs of black-box function for each input
            l: length parameter for kernel
            sigma_f: standard deviation given to output of black-box function
        """
        self.X = X_init
        self.Y = Y_init
        self.l = l
        self.sigma_f = sigma_f
        self.K = self.kernel(X_init, X_init)

    def kernel(self, X1, X2):
        """ calculates covariance kernel matrix between two matrices

        Arguments:
            X1: np arr (m, 1)
            X2: np arr (n, 1)

        Returns: covariance kernel matrix between X1 and X2
        """
        sqdist = np.sum(X1 ** 2, 1).reshape(-1, 1) +\
            np.sum(X2 ** 2, 1) - 2 * np.matmul(X1, X2.T)

        return self.sigma_f ** 2 * np.exp(-0.5 / self.l ** 2 * sqdist)
