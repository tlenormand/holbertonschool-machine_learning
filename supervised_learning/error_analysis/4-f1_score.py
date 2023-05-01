#!/usr/bin/env python3

import numpy as np


def f1_score(confusion):
    """ calculates the F1 score of a confusion matrix

    Arguments:
        confusion (numpy.ndarray): confusion matrix of shape (classes, classes)
            where row indices represent the correct labels
            and column indices represent the predicted labels
    Returns:
        numpy.ndarray: F1 score of each class
    """
    # F1 = 2 * (precision * sensitivity) / (precision + sensitivity)
    precision = np.zeros(confusion.shape[0])
    sensitivity = np.zeros(confusion.shape[0])

    for i in range(confusion.shape[0]):
        true_positives = confusion[i, i]
        all_positives = np.sum(confusion[i])
        precision[i] = true_positives / all_positives

        all_positives = np.sum(confusion[:, i])
        sensitivity[i] = true_positives / all_positives

    return 2 * (precision * sensitivity) / (precision + sensitivity)
