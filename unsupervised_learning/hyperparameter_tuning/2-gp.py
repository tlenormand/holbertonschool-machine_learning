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

    def predict(self, X_s):
        """ predicts mean, standard deviation of points in Gaussian process

        Arguments:
            X_s: np arr (s, 1) of points whose mean, std dev will be calculated

        Returns: mu, sigma
            mu: np arr (s,) of mean for each point in X_s
            sigma: np arr (s,) of std dev for each point in X_s
        """
        K_inv = np.linalg.inv(self.K)
        K_s = self.kernel(self.X, X_s)
        K_ss = self.kernel(X_s, X_s)
        K_sT_K_inv = np.matmul(K_s.T, K_inv)
        mu = np.matmul(K_sT_K_inv, self.Y).reshape(-1)
        cov = K_ss - np.matmul(K_sT_K_inv, K_s)
        sigma = np.diagonal(cov)

        return mu, sigma

    def update(self, X_new, Y_new):
        """ updates Gaussian Process

        Arguments:
            X_new: np arr (1,) new sample input
            Y_new: np arr (1,) new sample output
        """
        self.X = np.concatenate((self.X, X_new.reshape((1, 1))), axis=0)
        self.Y = np.concatenate((self.Y, Y_new.reshape((1, 1))), axis=0)
        self.K = self.kernel(self.X, self.X)
