#!/usr/bin/env python3
""" Moving Average """

import numpy as np


def moving_average(data, beta):
    """ Calculates the weighted moving average of a data set

    Argumets:
        data is the list of data to calculate the moving average of
        beta is the weight used for the moving average

    Returns:
        a list containing the moving averages of data
    """
    # formula of moving average is: Vt = beta * Vt-1 + (1 - beta) * Xt
    vt = 0
    moving_averages = []

    for i in range(len(data)):
        vt = beta * vt + (1 - beta) * data[i]
        moving_averages.append(vt / (1 - beta ** (i + 1)))

    return moving_averages
