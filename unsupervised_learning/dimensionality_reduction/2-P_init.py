#!/usr/bin/env python3
""" Dimensionality Reduction """

import numpy as np


def P_init(X, perplexity):
    """ initializes all variables required to calculate
        the P affinities in t-SNE

    Arguments:
        X {np.ndarray} -- containing the dataset of shape (n, d)
            n is the number of data points
            d is the number of dimensions in each point
        perplexity {float} -- perplexity that all Gaussian distributions
            should have

    Returns:
        (D, P, betas, H)
        D {np.ndarray} - initialized pairwise distance of shape (n, n)
        P {np.ndarray} - initialized P affinities of shape (n, n)
        betas {np.ndarray} - initialized beta values of shape (n, 1)
        H {float} - Shannon entropy of the Shannon entropy of P
    """
    n, d = X.shape

    # calculate the squared pairwise distance between two data points
    sum_X = np.sum(np.square(X), axis=1)
    D = np.add(np.add(-2 * np.dot(X, X.T), sum_X).T, sum_X)

    # initialize P, betas, and H
    P = np.zeros((n, n))
    betas = np.ones((n, 1))
    H = np.log2(perplexity)

    return (D, P, betas, H)
