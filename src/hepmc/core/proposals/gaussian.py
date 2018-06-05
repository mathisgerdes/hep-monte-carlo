import numpy as np
from scipy.stats import multivariate_normal as multi_norm

from ..density import Proposal


class Gaussian(Proposal):

    def __init__(self, ndim=1, cov=None, scale=None):
        super().__init__(ndim, True)
        self._cov = None

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

    def proposal(self, state):
        sample = np.random.multivariate_normal(state, self.cov)
        return sample

    def proposal_pdf(self, state, candidate):
        prob = multi_norm.pdf(candidate, state, self.cov)  # symmetric
        return prob

    @property
    def cov(self):
        return self._cov

    @cov.setter
    def cov(self, value):
        self._cov = np.eye(self.ndim, self.ndim) * np.asanyarray(value)
