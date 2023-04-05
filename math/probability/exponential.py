#!/usr/bin/env python3
"""
    Class Exponential : represents an exponential distribution
"""


class Exponential:
    """ Class Exponential that represents an exponential distribution

    Attributes:
        pi: value of pi
        e: value of e
        lambtha: expected number of occurences in a given time frame
        data: list of the data to be used to estimate the distribution

    Methods:
        pdf: Calculates the value of the PDF for a given time period
        cdf: Calculates the value of the CDF for a given time period
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

        if data is None:
            if lambtha <= 0:
                raise ValueError("lambtha must be a positive value")
        else:
            if not isinstance(self.data, list):
                raise TypeError("data must be a list")
            if len(self.data) < 2:
                raise ValueError("data must contain multiple values")

            # formula for lambtha is 1 / (sum of data / length of data)
            self.lambtha = float(1 / (sum(self.data) / len(self.data)))

    def pdf(self, x):
        """ Calculates the value of the PDF for a given time period

        Arguments:
            x: time period

        Returns:
            PDF value for x
        """
        if x < 0:
            return 0

        # formula for PDF is lambtha * e^(-lambtha * x)
        return self.lambtha * self.e ** (-self.lambtha * x)

    def cdf(self, x):
        """ Calculates the value of the CDF for a given time period

        Arguments:
            x: time period

        Returns:
            CDF value for x
        """
        if x < 0:
            return 0

        # formula for CDF is 1 - e^(-lambtha * x)
        return 1 - self.e ** (-self.lambtha * x)
