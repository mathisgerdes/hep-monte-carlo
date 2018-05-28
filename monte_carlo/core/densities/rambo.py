import numpy as np
from math import gamma
from ..density import Distribution
from ..phase_space import map_rambo


class Rambo(Distribution):

    def __init__(self, nparticles, E_CM):
        ndim = 4*nparticles
        super().__init__(ndim, False)

        self.nparticles = nparticles
        self.E_CM = E_CM

    def rvs(self, sample_size):
        xs = np.random.random((sample_size, self.ndim))
        return map_rambo(xs, self.E_CM, self.nparticles)

    def pdf(self, xs):
        return (np.pi/2.)**(self.nparticles-1) * self.E_CM**(2*self.nparticles-4)/gamma(self.nparticles)/gamma(self.nparticles-1)
