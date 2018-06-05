from ..density import Density
from ..util import hypercube_bounded
import numpy as np


class MappedDensity(Density):

    def __init__(self, density, mapping, norm=None):
        super().__init__(mapping.ndim)
        self.density = density
        self.mapping = mapping
        self.norm = norm

    @hypercube_bounded(1, self_has_ndim=True)
    def pdf(self, xs):
        ps = self.mapping.map(xs)
        pdf = self.density.pdf(ps) / self.mapping.pdf(xs)
        if self.norm is not None:
            return pdf / self.norm
        return pdf

    @hypercube_bounded(1, self_has_ndim=True)
    def pdf_gradient(self, xs):
        raise NotImplementedError  # mapping pdf grad. generally not known

    @hypercube_bounded(1, self_has_ndim=True)
    def pot(self, xs):
        ps = self.mapping.map(xs)
        pot = self.density.pot(ps)
        return pot - np.log(self.norm) - np.log(self.mapping.pdf(xs))

    @hypercube_bounded(1, self_has_ndim=True)
    def pot_gradient(self, xs):
        raise NotImplementedError
