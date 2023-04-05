#!/usr/bin/env python3
"""
    Class Binomial : represents a binomial distribution
"""


class Binomial:
    """ Class Binomial that represents a binomial distribution

    Attributes:
        pi: value of pi
        e: value of e
        lambtha: expected number of occurences in a given time frame
        data: list of the data to be used to estimate the distribution
        n: number of Bernoulli trials
        p: probability of a “success”

    Methods:

    """
    def __init__(self, data=None, n=1, p=0.5):
        """ Constructor

        Arguments:
            data: list of the data to be used to estimate the distribution
            n: number of Bernoulli trials
            p: probability of a “success”
        """
        self.pi = 3.1415926536
        self.e = 2.7182818285
        self.p = float(p)
        self.n = int(n)
        self.data = data

        if data is None:
            if n <= 0:
                raise ValueError("n must be a positive value")

            if p <= 0 or p >= 1:
                raise ValueError("p must be greater than 0 and less than 1")
        else:
            if not isinstance(self.data, list):
                raise TypeError("data must be a list")

            if len(self.data) < 2:
                raise ValueError("data must contain multiple values")

            self.mean = sum(self.data) / len(self.data)
            self.stddev = (sum([(x - self.mean) ** 2 for x in self.data]) /
                           len(self.data))
            self.p = 1 - (self.stddev / self.mean)
            self.n = round(self.mean / self.p)
            self.p = self.mean / self.n

    def pmf(self, k):
        """ Calculates the value of the PMF for a given number of “successes”

        Arguments:
            k: number of “successes”

        Returns:
            PMF value for k
        """
        k = int(k)

        if k < 0:
            return 0

        n_fac = 1
        k_fac = 1
        n_k_fac = 1

        for i in range(1, self.n + 1):
            n_fac *= i

            if i == k:
                k_fac = n_fac
    
            if i == self.n - k:
                n_k_fac = n_fac

        return (n_fac / (k_fac * n_k_fac) *
                self.p ** k * (1 - self.p) ** (self.n - k))

    def cdf(self, k):
        """ Calculates the value of the CDF for a given number of “successes”

        Arguments:
            k: number of “successes”

        Returns:
            CDF value for k
        """
        k = int(k)

        if k < 0:
            return 0

        cdf = 0
        for i in range(k + 1):
            cdf += self.pmf(i)

        return cdf
