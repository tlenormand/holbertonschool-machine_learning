#!/usr/bin/env python3
""" Adam """

import tensorflow as tf


def update_variables_Adam(alpha, beta1, beta2, epsilon, var, grad, v, s, t):
    """ Updates a variable in place using the Adam optimization algorithm.

        Args:
            alpha: (float) the learning rate
            beta1: (float) the weight used for the first moment
            beta2: (float) the weight used for the second moment
            epsilon: (float) a small number to avoid division by zero
            var: (numpy.ndarray) containing the variable to be updated
            grad: (numpy.ndarray) containing the gradient of var
            v: (float) the previous first moment of var
            s: (float) the previous second moment of var
            t: (int) the time step used for bias correction

        Returns:
            The updated variable, the new first moment, and the new second
            moment, respectively.
    """
    # formula for Adam is:
    # Vdw = beta1 * Vdw + (1 - beta1) * dW => momentum
    # Vdw_corrected = Vdw / (1 - beta1**t) => momentum

    # Sdw = beta2 * Sdw + (1 - beta2) * dW**2 => RMSProp
    # Sdw_corrected = Sdw / (1 - beta2**t) => RMSProp

    # W = W - alpha * Vdw_corrected / (Sdw_corrected**0.5 + epsilon) => Adam

    vdw = beta1 * v + (1 - beta1) * grad
    vdw_corrected = vdw / (1 - beta1**t)

    sdw = beta2 * s + (1 - beta2) * grad ** 2
    sdw_corrected = sdw / (1 - beta2 ** t)

    w = var - alpha * vdw_corrected / (sdw_corrected ** 0.5 + epsilon)

    return w, vdw, sdw
