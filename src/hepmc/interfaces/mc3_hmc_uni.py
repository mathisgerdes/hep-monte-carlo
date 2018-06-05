import numpy as np
from ..core import densities, DefaultMetropolis, MixingMarkovUpdate,\
    MultiChannelMC, MultiChannel
from ..hamiltonian import HamiltonianUpdate


def get_sampler(target, ndim, initial, beta, mass=10, steps=10, step_size=.1):

    if np.isscalar(initial):
        initial = np.full(ndim, initial)

    # importance
    is_sampler = DefaultMetropolis(ndim, target)

    momentum_dist = densities.Gaussian(ndim, cov=mass)
    local_sampler = HamiltonianUpdate(
        target, momentum_dist, steps, step_size)

    updates = [is_sampler, local_sampler]
    sampler = MixingMarkovUpdate(ndim, updates, [beta, 1 - beta])

    return sampler, initial
