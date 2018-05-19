from ..density import Density
from scipy.stats import multivariate_normal as multi_norm

import numpy as np

from ..util import interpret_array, hypercube_bounded


class UnconstrainedCamel(Density):
    """
    sum of two gaussians on a [0, 1] hypercube
    """
    def __init__(self, ndim, mu_a=1/3, mu_b=2/3, a=0.1):
        super().__init__(ndim, False)
        self._mu_a = self._mu_b = None
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.cov = a**2/2
    
    def pdf(self, xs):
        xs = interpret_array(xs, self.ndim)
        ndim = xs.shape[1]

        if ndim != self.ndim:
            raise RuntimeWarning("Mismatching dimensions.")

        pdf_a = np.atleast_1d(
            multi_norm.pdf(xs, mean=self.mu_a, cov=self.cov))
        pdf_b = np.atleast_1d(
            multi_norm.pdf(xs, mean=self.mu_b, cov=self.cov))
        return (pdf_a + pdf_b).flatten() / 2
    
    def pdf_gradient(self, xs):
        xs = interpret_array(xs, self.ndim)
        ndim = xs.shape[1]

        if ndim != self.ndim:
            raise RuntimeWarning("Mismatching dimensions.")

        pdf_a = np.atleast_1d(
            multi_norm.pdf(xs, mean=self.mu_a, cov=self.cov))
        pdf_b = np.atleast_1d(
            multi_norm.pdf(xs, mean=self.mu_b, cov=self.cov))
        return ((-xs + self.mu_a) / self.cov * pdf_a[:, np.newaxis] +
                (-xs + self.mu_b) / self.cov * pdf_b[:, np.newaxis]) / 2

    @property
    def mean(self):
        return (self.mu_a + self.mu_b) / 2

    @property
    def variance(self):
        # (var_a + var_b) / 2 + (mu_a - mu_b)^2 / 4
        return self.cov + (self.mu_a - self.mu_b) ** 2

    @property
    def mu_a(self):
        return self._mu_a

    @mu_a.setter
    def mu_a(self, value):
        if np.isscalar(value):
            self._mu_a = np.full(self.ndim, value)
        else:
            self._mu_a = np.array(value, copy=False, subok=True, ndmin=1)

    @property
    def mu_b(self):
        return self._mu_b

    @mu_b.setter
    def mu_b(self, value):
        if np.isscalar(value):
            self._mu_b = np.full(self.ndim, value)
        else:
            self._mu_b = np.array(value, copy=False, subok=True, ndmin=1)

    def __repr__(self):
        return type(self).__name__ + "(ndim=%s, mu_a=%s, mu_b=%s, a=%s)" % (
            self.ndim, self.mu_a, self.mu_b, np.sqrt(2*self.cov))


class Camel(UnconstrainedCamel):
    """
    sum of two gaussians on a [0, 1] hypercube
    """
    @hypercube_bounded(1, self_has_ndim=True)
    def pdf(self, xs):
        return super().pdf(xs).flatten()

    def pdf_gradient(self, xs):
        xs = interpret_array(xs, self.ndim)

        in_bounds = np.all((0 < xs) * (xs < 1), axis=1)

        res = np.empty(xs.shape)
        res[in_bounds] = super().pdf_gradient(xs[in_bounds])
        res[np.logical_not(in_bounds)] = 0

        return res