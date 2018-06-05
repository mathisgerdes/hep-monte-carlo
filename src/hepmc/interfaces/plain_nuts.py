import numpy as np
from ..core.densities import Gaussian
from ..hamiltonian import NUTSUpdate


def get_sampler(target, ndim, initial, mass=10, nadapt=100):
    if np.isscalar(initial):
        initial = np.full(ndim, initial)

    momentum_dist = Gaussian(ndim, cov=mass)
    sampler = NUTSUpdate(target, momentum_dist, lambda t: t <= nadapt)

    return sampler, initial
