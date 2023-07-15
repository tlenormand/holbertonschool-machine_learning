#!/usr/bin/env python3
""" sklearn """

import sklearn.cluster


def kmeans(X, k):
    """
    function that performs K-means on a dataset
    Args:
        X: numpy.ndarray of shape (n, d) containing the dataset
        k: number of clusters
    Return: C, clss
        C: numpy.ndarray of shape (k, d) containing
            the centroid means for each cluster
        clss: numpy.ndarray of shape (n,) containing
            the index of the cluster in C that each data point belongs to
    """
    k_means = sklearn.cluster.KMeans(n_clusters=k).fit(X)
    C = k_means.cluster_centers_
    clss = k_means.labels_

    return C, clss
