import numpy as np

from .base import Distribution
from ..util import interpret_array


class Uniform(Distribution):

    def __init__(self, ndim, sample_range=(0, 1)):
        super().__init__(ndim, True)
        self.sample_min = sample_range[0]
        self.sample_max = sample_range[1]
        self.sample_diff = sample_range[1] - sample_range[0]
        self.vol = np.prod(self.sample_diff)

    def pdf(self, xs):
        xs = interpret_array(xs, self.ndim)
        density = (np.all(xs > self.sample_min, axis=1) *
                   np.all(xs < self.sample_max, axis=1) / self.vol)
        return density

    def pdf_gradient(self, xs):
        return np.zeros(xs.shape)

    def pot(self, xs):
        xs = interpret_array(xs, self.ndim)
        logpdf = -(np.all(xs > self.sample_min, axis=1) *
                   np.all(xs < self.sample_max, axis=1) * np.log(self.vol))
        return logpdf

    def pot_gradient(self, xs):
        return np.zeros(xs.shape)

    def rvs(self, sample_size):
        rnd = np.random.random((sample_size, self.ndim))
        return self.sample_min + self.sample_diff * rnd
