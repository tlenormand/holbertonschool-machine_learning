#!/usr/bin/env python3
"""
    Class Poisson : represents a poisson distribution
"""


class Poisson:
    """ Class Poisson that represents a poisson distribution

    Attributes:
        pi: value of pi
        e: value of e
        lambtha: expected number of occurences in a given time frame
        data: list of the data to be used to estimate the distribution

    Methods:
        pmf: Calculates the value of the PMF for a given number of successes
        cdf: Calculates the value of the CDF for a given number of successes
    """
    def __init__(self, data=None, lambtha=1.):
        """ Constructor

        Arguments:
            data: list of the data to be used to estimate the distribution
            lambtha: expected number of occurences in a given time frame
        """
        self.pi = 3.1415926536
        self.e = 2.7182818285
        self.data = data
        self.lambtha = float(lambtha)

        if self.data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(self.data, list):
                raise TypeError("data must be a list")
            if len(self.data) < 2:
                raise ValueError("data must contain multiple values")

            # formula for lambtha is sum of data / length of data
            self.lambtha = float(sum(self.data) / len(self.data))

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of successes

        Arguments:
            k: number of successes

        Returns:
            PMF value for k
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        k_factorial = 1
        for i in range(1, k + 1):
            k_factorial *= i

        # formula for PMF is (e^(-lambtha) * lambtha^k) / k!
        return (self.lambtha ** k * self.e ** (-self.lambtha)) / k_factorial

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of successes

        Arguments:
            k: number of successes

        Returns:
            CDF value for k
        """
        if not isinstance(k, int):
            k = int(k)

        if k < 0:
            return 0

        cdf = 0
        for i in range(k + 1):
            # formula for CDF is sum of PMF from 0 to k
            cdf += self.pmf(i)

        return cdf
