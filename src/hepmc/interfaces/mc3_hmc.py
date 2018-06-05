import numpy as np
from ..core import densities, DefaultMetropolis, MixingMarkovUpdate,\
    MultiChannelMC, MultiChannel
from ..hamiltonian import HamiltonianUpdate


def get_sampler(target, ndim, initial, centers, widths, beta,
                nintegral=1000, mass=10, steps=10, step_size=.1):

    if np.isscalar(initial):
        initial = np.full(ndim, initial)

    # importance
    channels = MultiChannel(
        [densities.Gaussian(ndim, mu=center, scale=width)
         for center, width in zip(centers, widths)])
    mc_importance = MultiChannelMC(channels)
    integration_sample = mc_importance(target, [], [nintegral], [])

    is_sampler = DefaultMetropolis(ndim, target, channels)

    momentum_dist = densities.Gaussian(ndim, cov=mass)
    local_sampler = HamiltonianUpdate(
        target, momentum_dist, steps, step_size)

    updates = [is_sampler, local_sampler]
    sampler = MixingMarkovUpdate(ndim, updates, [beta, 1 - beta])

    return sampler, initial
