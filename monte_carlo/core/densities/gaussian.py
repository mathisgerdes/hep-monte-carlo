import numpy as np
from scipy.stats import multivariate_normal as multi_norm

from ..density import Distribution
from ..util import interpret_array


class Gaussian(Distribution):

    def __init__(self, ndim, mu=0, cov=None, scale=None):
        super().__init__(ndim, False)

        self._mean = None
        self._cov = None
        self._cov_inv = None
        self.mean = mu

        if cov is None:
            if scale is None:
                self.cov = np.ones(ndim)
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
        prob = multi_norm.pdf(xs, self.mean, self.cov)
        return np.array(prob, ndmin=1, copy=False, subok=True)

    def pdf_gradient(self, xs):
        xs = interpret_array(xs, self.ndim)
        return - (np.dot(self._cov_inv, (xs - self.mean)) *
                  self.pdf(xs)[:, np.newaxis])

    def pot(self, xs):
        xs = interpret_array(xs, self.ndim)
        logpdf = -multi_norm.logpdf(xs, self.mean, self.cov)
        return np.array(logpdf, copy=False, ndmin=1, subok=True)

    def pot_gradient(self, xs):
        xs = interpret_array(xs, self.ndim)
        return np.einsum('ij,kj->ki', self._cov_inv, xs - self.mean)

    def rvs(self, sample_size):
        sample = np.random.multivariate_normal(self.mean, self.cov, sample_size)
        return sample

    @property
    def mean(self):
        return self._mean

    @mean.setter
    def mean(self, value):
        if np.isscalar(value):
            self._mean = np.empty(self.ndim)
            self._mean.fill(value)
        else:
            self._mean = np.array(value, copy=False, subok=True, ndmin=1)

    @property
    def variance(self):
        return np.diagonal(self._cov)

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, value):
        self._cov = np.eye(self.ndim, self.ndim) * np.asanyarray(value)
        self._cov_inv = np.linalg.inv(self._cov)

    def __repr__(self):
        return type(self).__name__ + "(ndim=%s, mu=%s, cov=%s)" % (
            self.ndim, self.mean, self.cov)
