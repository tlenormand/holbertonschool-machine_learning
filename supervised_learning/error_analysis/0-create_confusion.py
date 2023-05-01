#!/usr/bin/env python3

import numpy as np


def create_confusion_matrix(labels, logits):
    """ Creates a confusion matrix

    Argumemts:
        labels (numpy.ndarray): contains the correct labels for each data point
        logits (numpy.ndarray): contains the predicted labels

    Returns:
        numpy.ndarray: confusion matrix
    """
    return np.matmul(labels.T, logits)
