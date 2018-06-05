import numpy as np
from ..hamiltonian import SphericalNUTS


def get_sampler(target, ndim, initial, nadapt=100):
    if np.isscalar(initial):
        initial = np.full(ndim, initial)

    sampler = SphericalNUTS(target, lambda t: t <= nadapt)

    return sampler, initial


