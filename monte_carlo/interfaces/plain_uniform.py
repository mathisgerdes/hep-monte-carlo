from ..core import DefaultMetropolis
import numpy as np


def get_sampler(target, ndim, initial):
    if np.isscalar(initial):
        initial = np.full(ndim, initial)
    return DefaultMetropolis(ndim, target), initial
