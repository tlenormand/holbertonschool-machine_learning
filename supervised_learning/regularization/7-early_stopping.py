#!/usr/bin/env python3
""" Early Stopping """

import numpy as np


def early_stopping(cost, opt_cost, threshold, patience, count):
    """ determines if you should stop gradient descent early

    Arguments:
        cost: current validation cost of the neural network
        opt_cost: lowest recorded validation cost of the neural network
        threshold: threshold used for early stopping
        patience: patience count used for early stopping
        count: count of how long the threshold has not been met

    Returns:
        boolean of whether the network should be stopped early, followed by
            the updated count
    """
    # if the threshold is not met, reset the count
    if opt_cost - cost > threshold:
        count = 0
    else:
        count += 1

    return (count == patience, count)
