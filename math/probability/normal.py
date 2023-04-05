#!/usr/bin/env python3
"""
    Class Normal : represents a normal distribution
"""


class Normal:
    """ Class Normal that represents a normal distribution

    Attributes:
        pi: value of pi
        e: value of e
        lambtha: expected number of occurences in a given time frame
        data: list of the data to be used to estimate the distribution

    Methods:
        pdf: Calculates the value of the PDF for a given time period
        cdf: Calculates the value of the CDF for a given time period
        z_score: Calculates the z-score of a given x-value
        x_value: Calculates the x-value of a given z-score
    """
    def __init__(self, data=None, mean=0., stddev=1.):
        """ Constructor

        Arguments:
            data: list of the data to be used to estimate the distribution
            mean: mean of the distribution
            stddev: standard deviation of the distribution
        """
        self.pi = 3.1415926536
        self.e = 2.7182818285
        self.data = data
        self.mean = float(mean)
        self.stddev = float(stddev)

        if data is None:
            if stddev <= 0:
                raise ValueError("stddev must be a positive value")
        else:
            if not isinstance(self.data, list):
                raise TypeError("data must be a list")
            if len(self.data) < 2:
                raise ValueError("data must contain multiple values")

            # formula for mean is sum of data / length of data
            self.mean = float(sum(self.data) / len(self.data))

            # formula for stddev is sqrt(sum of (x - mean)^2 / length of data)
            self.stddev = float((sum([
                (x - self.mean) ** 2 for x in self.data
                ]) / len(self.data)) ** 0.5)

    def z_score(self, x):
        """ Calculates the z-score of a given x-value

        Arguments:
            x: x-value

        Returns:
            z-score of x
        """
        # formula for z-score is (x - mean) / stddev
        return (x - self.mean) / self.stddev

    def x_value(self, z):
        """ Calculates the x-value of a given z-score

        Arguments:
            z: z-score

        Returns:
            x-value of z
        """
        # formula for x-value is stddev * z + mean
        return self.stddev * z + self.mean

    def pdf(self, x):
        """ Calculates the value of the PDF for a given time period

        Arguments:
            x: time period

        Returns:
            PDF value for x
        """
        # formula for PDF is:
        # (1 / (stddev * sqrt(2 * pi))) * e^(-1/2 * ((x - mean) / stddev)^2)
        return (1 / (self.stddev * (2 * self.pi) ** 0.5)) *\
            self.e ** (-0.5 * ((x - self.mean) / self.stddev) ** 2)

    def cdf(self, x):
        """ Calculates the value of the CDF for a given time period

        Arguments:
            x: time period

        Returns:
            CDF value for x
        """
        # formula for CDF is:
        # (1 / 2) * (1 + erf((x - mean) / (stddev * sqrt(2))))
        return 0.5 * (1 + self._erf((x - self.mean) /
                                    (self.stddev * (2 ** 0.5))))

#####################
# Private functions #
#####################

    def _erf(self, x):
        """ Calculates the error function

        Arguments:
            x: x-value

        Returns:
            error function value for x
        """
        return ((2 / self.pi ** 0.5) * (
            x - (x ** 3 / 3) +
            (x ** 5 / 10) -
            (x ** 7 / 42) +
            (x ** 9 / 216)
        ))
