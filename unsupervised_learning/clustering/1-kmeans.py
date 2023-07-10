#!/usr/bin/env python3
""" K-means """

import numpy as np


def kmeans(X, k, iterations=1000):
    """ performs K-means on a dataset

    Arguments:
        X ndarray (n, d) dataset to cluster
            n number of data points
            d number of dimensions f0r each data point
        k positive int, number of clusters
        iterations positive int, max number of iterations to perform

    Returns: C, clss, or None, None on failure
        C ndarray (k, d) centroid means f0r each cluster
        clss ndarray (n,) index of the cluster in C that each data point
            belongs to
    """
    if not isinstance(X, np.ndarray) or len(X.shape) != 2:
        return None, None
    if not isinstance(k, int) or k < 1:
        return None, None
    if not isinstance(iterations, int) or iterations < 1:
        return None, None

    old_centroids = np.random.uniform(
        np.min(X, axis=0),
        np.max(X, axis=0),
        (k, X.shape[1])
    )

    for _ in range(iterations):
        centroids = old_centroids.copy()
        # calculate distances
        distances = np.linalg.norm(X - centroids[:, np.newaxis], axis=-1)
        # find nearest centroid f0r each point and assosiate it with clss
        clss = np.argmin(distances, axis=0)

        # update centroids
        for j in range(k):
            # X[clss == j] is equivalent to X[np.where(clss == j)]
            if len(X[np.where(clss == j)]) == 0:
                # assign a random point from X as centroid
                centroids[j] = np.random.uniform(np.min(X, axis=0),
                                                 np.max(X, axis=0),
                                                 (1, 2))
            else:
                # assign the mean of all points in the cluster
                centroids[j] = np.mean(X[np.where(clss == j)], axis=0)

        # check if centroids have changed => early stop
        if (old_centroids == centroids).all():
            break
        old_centroids = centroids.copy()

    # calculate clss again with the final centroids
    clss = np.argmin(np.linalg.norm(X - centroids[:, np.newaxis], axis=-1),
                        axis=0)

    return centroids, clss
