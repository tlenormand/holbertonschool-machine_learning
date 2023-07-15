#!/usr/bin/env python3
""" Agglomerative """

import scipy.cluster.hierarchy


def agglomerative(X, dist):
    """ performs agglomerative clustering on a dataset

    Arguments:
        X np.ndarray of shape (n, d) containing the dataset
        dist maximum cophenetic distance for all clusters

    Returns:
        clss np.ndarray shape (n,) containing the cluster indices for each data
            point
    """
    Z = scipy.cluster.hierarchy.linkage(X, method='ward')
    clss = scipy.cluster.hierarchy.fcluster(Z, dist, criterion='distance')

    return clss
