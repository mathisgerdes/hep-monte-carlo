import numpy as np
from ..density import Distribution
from ..phase_space import map_rambo, map_rambo_on_diet, rambo_pdf


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
        return rambo_pdf(xs, self.E_CM, self.nparticles)


class RamboOnDiet(Distribution):

    def __init__(self, nparticles, E_CM):
        ndim = 3*nparticles-4
        super().__init__(ndim, False)

        self.nparticles = nparticles
        self.E_CM = E_CM

    def rvs(self, sample_size):
        xs = np.random.random((sample_size, self.ndim))
        return map_rambo_on_diet(xs, self.E_CM, self.nparticles)

    def pdf(self, xs):
        return rambo_pdf(xs, self.E_CM, self.nparticles)
