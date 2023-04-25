#!/usr/bin/env python3
""" RMSProp """


def update_variables_RMSProp(alpha, beta2, epsilon, var, grad, s):
    """ Updates a variable using the RMSProp optimization algorithm.

        Args:
            alpha: (float) the learning rate
            beta2: (float) the RMSProp weight
            epsilon: (float) a small number to avoid division by zero
            var: (numpy.ndarray) containing the variable to be updated
            grad: (numpy.ndarray) containing the gradient of var
            s: (float) the previous second moment of var

        Returns:
            The updated variable and the new moment, respectively.
    """
    # formula for RMSProp is:
    # Sdw = beta2 * Sdw + (1 - beta2) * dW**2
    # W = W - alpha * dW / sqrt(Sdw + epsilon)
    sdw = beta2 * s + (1 - beta2) * grad ** 2
    w = var - alpha * grad / (sdw ** 0.5 + epsilon)

    return w, sdw
