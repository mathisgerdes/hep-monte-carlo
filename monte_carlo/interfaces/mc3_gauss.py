import numpy as np
from ..core import proposals, densities, DefaultMetropolis, MixingMarkovUpdate,\
    MultiChannelMC, MultiChannel


def get_sampler(target, ndim, initial, centers, widths, beta,
                nintegral=1000, local_width=.1):
    if np.isscalar(initial):
        initial = np.full(ndim, initial)

    # importance
    channels = MultiChannel(
        [densities.Gaussian(ndim, mu=center, scale=width)
         for center, width in zip(centers, widths)])
    mc_importance = MultiChannelMC(channels)
    integration_sample = mc_importance(target, [], [nintegral], [])

    is_sampler = DefaultMetropolis(ndim, target, channels)

    local_proposal = proposals.Gaussian(ndim, scale=local_width)
    local_sampler = DefaultMetropolis(ndim, target, local_proposal)

    updates = [is_sampler, local_sampler]
    sampler = MixingMarkovUpdate(ndim, updates, [beta, 1 - beta])

    return sampler, initial
