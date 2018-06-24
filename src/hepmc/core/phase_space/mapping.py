from ..density import Density
from ..util import hypercube_bounded
from ..markov import MarkovUpdate
import numpy as np


class PhaseSpaceMapping(object):
    def __init__(self, ndim, ndim_out=None):
        self.ndim = ndim  # input dimensionality
        self.ndim_out = ndim_out

    def pdf(self, xs):
        raise NotImplementedError

    def pdf_gradient(self, xs):
        raise NotImplementedError

    def map(self, xs):
        raise NotImplementedError

    def map_inverse(self, xs):
        raise NotImplementedError

    def input_remapper(self, other):
        """ this returns mapping: xs -> other^-1 ( self ( xs ) ) """
        class Mapping(PhaseSpaceMapping):
            def pdf(self, xs):
                return other.map_inverse(self.map(xs))

            def pdf_gradient(self, xs):
                return other.pdf_gradient(self.map_inverse(xs))

            def map(self, xs):
                return other.map_inverse(self.map(xs))

            def map_inverse(self, xs):
                return other.pdf(self.map_inverse(xs))


class MappedDensity(Density):
    """
    Map the inputs before they are passed to the density
    """
    def __init__(self, density, mapping, norm=1):
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

    def pot_gradient(self, xs):
        ps = self.mapping.map(xs)
        grad = self.density.pot_gradient(ps)
        return grad - self.mapping.pdf_gradient(xs) / self.mapping.pdf(xs)

# class MappedMarkov(MarkovUpdate):
#
#     def __init__(self, update, map):