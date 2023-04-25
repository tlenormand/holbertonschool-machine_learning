#!/usr/bin/env python3
""" Momentum """

import numpy as np


def update_variables_momentum(alpha, beta1, var, grad, v):
    """ Updates a variable using the gradient descent with momentum
        optimization algorithm.

        Args:
            alpha: (float) the learning rate
            beta1: (float) the momentum weight
            var: (numpy.ndarray) containing the variable to be updated
            grad: (numpy.ndarray) containing the gradient of var
            v: (float) the previous first moment of var

        Returns:
            The updated variable and the new moment, respectively.
    """
    # formula for momentum is:
    # Vdw = beta1 * Vdw + (1 - beta1) * dW
    # W = W - alpha * Vdw
    vdw = beta1 * v + (1 - beta1) * grad
    w = var - alpha * vdw

    return w, vdw
