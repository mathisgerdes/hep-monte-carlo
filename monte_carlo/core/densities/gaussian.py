import numpy as np
from scipy.stats import multivariate_normal as multi_norm

from .base import Distribution
from ..util import interpret_array


class Gaussian(Distribution):

    def __init__(self, ndim, mu=0, cov=None, scale=None):
        super().__init__(ndim, True)

        self._mu = None
        self.mu = mu

        if cov is None:
            if scale is None:
                self.cov = 1
            else:
                if not np.isscalar(scale):
                    scale = np.atleast_1d(scale)
                self.cov = scale ** 2
        else:
            self.cov = cov
            if scale is not None:
                raise RuntimeWarning("Specified both cov and scale (using cov)")

    def pdf(self, xs):
        xs = interpret_array(xs, self.ndim)
        prob = multi_norm.pdf(xs, self.mu, self.cov)
        return np.array(prob, ndmin=1, copy=False, subok=True)

    def pdf_gradient(self, xs):
        xs = interpret_array(xs, self.ndim)
        return - (xs-self.mu) / self.cov * self.pdf(xs)[:, np.newaxis]

    def pot(self, xs):
        xs = interpret_array(xs, self.ndim)
        logpdf = -multi_norm.logpdf(xs, self.mu, self.cov)
        return np.array(logpdf, copy=False, ndmin=1, subok=True)

    def pot_gradient(self, xs):
        return np.atleast_1d(xs / self.cov)

    def rvs(self, sample_size):
        return np.random.normal(self.mu, self.cov, (sample_size, self.ndim))

    @property
    def mu(self):
        return self._mu

    @mu.setter
    def mu(self, value):
        if np.isscalar(value):
            self._mu = np.empty(self.ndim)
            self._mu.fill(value)
        else:
            self._mu = np.array(value, copy=False, subok=True, ndmin=1)
