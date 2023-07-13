#!/usr/bin/env python3
""" Expectation Maximization """

import numpy as np

initialize = __import__('4-initialize').initialize
expectation = __import__('6-expectation').expectation
maximization = __import__('7-maximization').maximization


def expectation_maximization(X, k, iterations=1000, tol=1e-5, verbose=False):
    """ performs the expectation maximization for a GMM

    Arguments:
        X np.ndarray (n, d) data set
        k positive int number of clusters
        iterations positive int number of iterations to perform
        tol non-negative float tolerance of log likelihood, used to determine
            early stopping i.e. if the difference is less than or equal to tol
            you should stop the algorithm
        verbose boolean that determines if you should print information about
            the algorithm

    Returns:
        pi, m, S, g, l, or None, None, None, None, None on failure
        pi np.ndarray (k,) priors for each cluster
        m np.ndarray (k, d) centroid means for each cluster
        S np.ndarray (k, d, d) covariance matrices for each cluster
        g np.ndarray (k, n) probabilities for each data point in each cluster
        log_l log likelihood of the model
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None, None, None, None
    if type(k) != int or k <= 0 or k >= X.shape[0]:
        return None, None, None, None, None
    if type(iterations) != int or iterations <= 0:
        return None, None, None, None, None
    if type(tol) != float or tol < 0:
        return None, None, None, None, None
    if type(verbose) != bool:
        return None, None, None, None, None

    # initialize pi, m, S, g, l
    pi, m, S = initialize(X, k)
    g, log_likelihood_initial = expectation(X, pi, m, S)
    i = 0

    while i < iterations:
        g, log_likelihood = expectation(X, pi, m, S)

        if verbose and i % 10 == 0:
            print("Log Likelihood after {} iterations: {}".format(
                i, round(log_likelihood, 5)))

        if abs(log_likelihood - log_likelihood_initial) <= tol:
            break

        pi, m, S = maximization(X, g)
        log_likelihood_initial = log_likelihood
        i += 1

    g, log_likelihood = expectation(X, pi, m, S)

    if verbose:
        print("Log Likelihood after {} iterations: {}".format(
            i, round(log_likelihood, 5)))

    return pi, m, S, g, log_likelihood
