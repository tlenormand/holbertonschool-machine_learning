#!/usr/bin/env python3
""" Error Analysis """

import numpy as np
precision = __import__('2-precision').precision
sensitivity = __import__('1-sensitivity').sensitivity


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
    return 2 * (precision(confusion) * sensitivity(confusion)) / \
        (precision(confusion) + sensitivity(confusion))
