from ..density import Density
from ..util import interpret_array
from scipy.stats import multivariate_normal
import numpy as np
from copy import copy


class Banana(Density):
    def __init__(self, ndim, bananicity=0.1):
        super().__init__(ndim, False)
        self.b = bananicity

    def pdf(self, xs):
        xs = interpret_array(xs, self.ndim)
        ndim = xs.shape[1]

        if ndim != self.ndim:
            raise RuntimeWarning("Mismatching dimensions.")

        phi = copy(xs)
        phi[:, 1] = phi[:, 1] + self.b*phi[:, 0]**2 - 100*self.b

        return multivariate_normal.pdf(phi, mean=np.zeros(ndim), cov=[100]+(ndim-1)*[1])

    def pot(self, xs):
        xs = interpret_array(xs, self.ndim)
        ndim = xs.shape[1]

        if ndim != self.ndim:
            raise RuntimeWarning("Mismatching dimensions.")

        phi = copy(xs)
        phi[:, 1] = phi[:, 1] + self.b*phi[:, 0]**2 - 100*self.b

        return -np.log(np.sqrt((2*np.pi)**ndim * 100)) - .5*(phi[:, 0]**2/100 + np.sum(phi[:, 1:]**2, axis=1))

    def pot_gradient(self, xs):
        xs = interpret_array(xs, self.ndim)
        ndim = xs.shape[1]

        if ndim != self.ndim:
            raise RuntimeWarning("Mismatching dimensions.")

        if ndim==2:
            y = np.empty_like(xs)
            y[:, 0] = -1/100*xs[:, 0]-2*self.b*xs[:, 0]*(xs[:, 1]+self.b*xs[:, 0]**2-100*self.b)
            y[:, 1] = 100*self.b-xs[:, 1]-self.b*xs[:, 0]**2
            return -y

        else:
            return NotImplementedError()

    def __repr__(self):
        return type(self).__name__ + "(ndim=%s, bananaicity=%s)" % (
            self.ndim, self.b)
