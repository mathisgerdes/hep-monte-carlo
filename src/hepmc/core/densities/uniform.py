import numpy as np

from ..density import Distribution
from ..util import interpret_array


class Uniform(Distribution):

    def __init__(self, ndim, sample_range=(0, 1)):
        super().__init__(ndim, True)
        self.low = sample_range[0]
        self.high = sample_range[1]
        self.vol = np.prod(np.diff(sample_range, axis=0))

    def pdf(self, xs):
        xs = interpret_array(xs, self.ndim)
        density = (np.all(xs > self.low, axis=1) *
                   np.all(xs < self.high, axis=1) / self.vol)
        return density

    def pdf_gradient(self, xs):
        return np.zeros(xs.shape)

    def pot(self, xs):
        xs = interpret_array(xs, self.ndim)
        logpdf = -(np.all(xs > self.low, axis=1) *
                   np.all(xs < self.high, axis=1) * np.log(self.vol))
        return logpdf

    def pot_gradient(self, xs):
        return np.zeros(xs.shape)

    def rvs(self, sample_size):
        return np.random.uniform(self.low, self.high, (sample_size, self.ndim))

    def __repr__(self):
        return type(self).__name__ + "(ndim=%s, sample_range=%s)" % (
            self.ndim, (self.low, self.high))
