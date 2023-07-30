#!/usr/bin/env python3
""" Bayesian Optimization """

import numpy as np
from scipy.stats import norm

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    """ performs Bayesian optimization on a noiseless 1D Gaussian process """
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """ constructor

        Arguments:
            f: black-box function to be optimized
            X_init: np arr (t, 1) inputs sampled with black-box function
            Y_init: np arr (t, 1) outputs of black-box function for each input
            bounds: tuple (min, max) representing bounds of the space to
                look for the optimal point
            ac_samples: number of samples that should be analyzed during
                acquisition
            l: length parameter for kernel
            sigma_f: standard deviation given to output of black-box function
            xsi: exploration-exploitation factor for acquisition
            minimize: bool determining whether optimization should be
                performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    #!/usr/bin/env python3
""" Bayesian Optimization """

import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """ constructor

        Arguments:
            f: black-box function to be optimized
            X_init: np arr (t, 1) inputs sampled with black-box function
            Y_init: np arr (t, 1) outputs of black-box function for each input
            bounds: tuple (min, max) representing bounds of the space to
                look for the optimal point
            ac_samples: number of samples that should be analyzed during
                acquisition
            l: length parameter for kernel
            sigma_f: standard deviation given to output of black-box function
            xsi: exploration-exploitation factor for acquisition
            minimize: bool determining whether optimization should be
                performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    #!/usr/bin/env python3
""" Bayesian Optimization """

import numpy as np

GP = __import__('2-gp').GaussianProcess


class BayesianOptimization:
    def __init__(self, f, X_init, Y_init, bounds, ac_samples, l=1, sigma_f=1,
                 xsi=0.01, minimize=True):
        """ constructor

        Arguments:
            f: black-box function to be optimized
            X_init: np arr (t, 1) inputs sampled with black-box function
            Y_init: np arr (t, 1) outputs of black-box function for each input
            bounds: tuple (min, max) representing bounds of the space to
                look for the optimal point
            ac_samples: number of samples that should be analyzed during
                acquisition
            l: length parameter for kernel
            sigma_f: standard deviation given to output of black-box function
            xsi: exploration-exploitation factor for acquisition
            minimize: bool determining whether optimization should be
                performed for minimization (True) or maximization (False)
        """
        self.f = f
        self.gp = GP(X_init, Y_init, l, sigma_f)
        self.X_s = np.linspace(bounds[0], bounds[1], ac_samples).reshape(-1, 1)
        self.xsi = xsi
        self.minimize = minimize

    def acquisition(self):
        """ calculates next best sample location

        uses the Expected Improvement acquisition function

        Returns: X_next, EI
            X_next: np arr (1,) representing next best sample point
            EI: np arr (ac_samples,) containing expected improvement of each
                potential sample
        """
        mu, sigma = self.gp.predict(self.X_s)
        mu_sample, _ = self.gp.predict(self.gp.X)

        if self.minimize is True:
            Y_s = np.min(mu_sample)
            imp = Y_s - mu - self.xsi
        else:
            Y_s = np.max(mu_sample)
            imp = mu - Y_s - self.xsi

        with np.errstate(divide='ignore'):
            Z = imp / sigma

        EI = imp * norm.cdf(Z) + sigma * norm.pdf(Z)
        for i in range(len(sigma)):
            if sigma[i] == 0.0:
                EI[i] = 0.0

        X_next = self.X_s[np.argmax(EI)]

        return X_next, EI
