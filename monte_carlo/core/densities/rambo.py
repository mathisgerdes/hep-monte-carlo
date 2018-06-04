import numpy as np
from ..density import Distribution
from .. import phase_space


class Rambo(Distribution):

    def __init__(self, nparticles, E_CM):
        self.mapping = phase_space.Rambo(E_CM, nparticles)
        super().__init__(self.mapping.ndim, False)

    def rvs(self, sample_size):
        xs = np.random.random((sample_size, self.ndim))
        return self.mapping.map(xs)

    def pdf(self, xs):
        return self.mapping.pdf(xs)

    @property
    def e_cm(self):
        return self.mapping.e_cm

    @e_cm.setter
    def e_cm(self, value):
        self.mapping.e_cm = value

    @property
    def nparticles(self):
        return self.mapping.nparticles

    @nparticles.setter
    def nparticles(self, value):
        self.mapping = phase_space.Rambo(self.e_cm, value)


class RamboOnDiet(Distribution):

    def __init__(self, nparticles, E_CM):
        self.mapping = phase_space.RamboOnDiet(E_CM, nparticles)
        super().__init__(self.mapping.ndim, False)

    def rvs(self, sample_size):
        xs = np.random.random((sample_size, self.ndim))
        return self.mapping.map(xs)

    def pdf(self, xs):
        return self.mapping.pdf(xs)

    @property
    def e_cm(self):
        return self.mapping.e_cm

    @e_cm.setter
    def e_cm(self, value):
        self.mapping.e_cm = value

    @property
    def nparticles(self):
        return self.mapping.nparticles

    @nparticles.setter
    def nparticles(self, value):
        self.mapping = phase_space.RamboOnDiet(self.e_cm, value)
