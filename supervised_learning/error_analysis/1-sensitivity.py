#!/usr/bin/env python3
""" Error Analysis """

import numpy as np


def sensitivity(confusion):
    """ Calculates the sensitivity for each class in a confusion matrix

    Argumemts:
        confusion (numpy.ndarray): confusion matrix of shape (classes, classes)
            where row indices represent the correct labels
            and column indices represent the predicted labels

    Returns:
        numpy.ndarray: sensitivity of each class
    """
    # sensitivity = true_positives / all_positives
    sensitivity = np.zeros(confusion.shape[0])

    for i in range(confusion.shape[0]):
        true_positives = confusion[i, i]
        all_positives = np.sum(confusion[i])

        sensitivity[i] = true_positives / all_positives

    return sensitivity
