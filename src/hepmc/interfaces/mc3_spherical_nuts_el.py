import numpy as np
from ..core import densities, DefaultMetropolis, MixingMarkovUpdate,\
    MultiChannelMC, MultiChannel, util
from ..hamiltonian import SphericalNUTS
from ..surrogate import extreme_learning


def get_sampler(target, ndim, initial, centers, widths, beta,
                nintegral=1000, nadapt=1000, el_size=100):

    if np.isscalar(initial):
        initial = np.full(ndim, initial)

    # importance
    channels = MultiChannel(
        [densities.Gaussian(ndim, mu=center, scale=width)
         for center, width in zip(centers, widths)])
    mc_importance = MultiChannelMC(channels)
    integration_sample = mc_importance(target, [], [nintegral], [])

    is_sampler = DefaultMetropolis(ndim, target, channels)

    # surrogate
    basis = extreme_learning.GaussianBasis(ndim)
    log_vals = -np.ma.log(integration_sample.function_values)
    xvals = integration_sample.data[~log_vals.mask]
    log_vals = log_vals[~log_vals.mask]
    # train
    params = basis.extreme_learning_train(xvals, log_vals, el_size)

    # surrogate gradient
    def surrogate_gradient(xs):
        return basis.eval_gradient(*params, xs)

    target.pot_gradient = surrogate_gradient
    util.count_calls(target, 'pot_gradient')

    # local sampler
    local_sampler = SphericalNUTS(target, lambda t: t <= nadapt)

    updates = [is_sampler, local_sampler]
    sampler = MixingMarkovUpdate(ndim, updates, [beta, 1 - beta])

    return sampler, initial
