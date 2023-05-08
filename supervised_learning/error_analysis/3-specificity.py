#!/usr/bin/env python3
""" Error Analysis """

import numpy as np


def specificity(confusion):
    """ Calculates the specificity for each class in a confusion matrix

    Argumemts:
        confusion (numpy.ndarray): confusion matrix of shape (classes, classes)
            where row indices represent the correct labels
            and column indices represent the predicted labels

    Returns:
        numpy.ndarray: specificity of each class
    """
    # specificity = true_negatives / all_negatives from the row
    specificity = np.zeros(confusion.shape[0])

    for i in range(confusion.shape[0]):
        true_negatives = np.sum(confusion) - np.sum(confusion[i]) - \
            np.sum(confusion[:, i]) + confusion[i][i]
        all_negatives = np.sum(confusion) - np.sum(confusion[i])

        specificity[i] = true_negatives / all_negatives

    return specificity
