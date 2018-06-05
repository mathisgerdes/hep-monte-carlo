import numpy as np
from ..hamiltonian import HamiltonianUpdate
from ..core.densities import Gaussian


def get_sampler(target, ndim, initial, mass=10, steps=10, step_size=.1):
    if np.isscalar(initial):
        initial = np.full(ndim, initial)

    momentum_dist = Gaussian(ndim, cov=mass)
    sampler = HamiltonianUpdate(
        target, momentum_dist, steps, step_size)

    return sampler, initial
