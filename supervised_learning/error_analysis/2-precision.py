#!/usr/bin/env python3
""" Error Analysis """

import numpy as np


def precision(confusion):
    """ Calculates the precision for each class in a confusion matrix

    Argumemts:
        confusion (numpy.ndarray): confusion matrix of shape (classes, classes)
            where row indices represent the correct labels
            and column indices represent the predicted labels

    Returns:
        numpy.ndarray: precision of each class
    """
    # precision = true_positives / all_positives from the column
    precision = np.zeros(confusion.shape[0])

    for i in range(confusion.shape[0]):
        true_positives = confusion[i][i]
        all_positives = np.sum(confusion[:][i])

        precision[i] = true_positives / all_positives

    return precision
