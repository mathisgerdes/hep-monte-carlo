import numpy as np


MINKOWSKI = np.diag([1, -1, -1, -1])


def map_fourvector_rambo(xs):
    """ Transform unit hypercube points into into four-vectors. """
    c = 2. * xs[:, :, 0] - 1.
    phi = 2. * np.pi * xs[:, :, 1]

    q = np.empty_like(xs)
    q[:, :, 0] = -np.log(xs[:, :, 2] * xs[:, :, 3])
    q[:, :, 1] = q[:, :, 0] * np.sqrt(1 - c ** 2) * np.cos(phi)
    q[:, :, 2] = q[:, :, 0] * np.sqrt(1 - c ** 2) * np.sin(phi)
    q[:, :, 3] = q[:, :, 0] * c

    return q


def map_rambo(xs, E_CM, nparticles=None):
    if nparticles is None:
        nparticles = xs.shape[1] // 4

    p = np.empty((xs.shape[0], nparticles, 4))

    q = map_fourvector_rambo(xs.reshape(xs.shape[0], nparticles, 4))
    # sum over all particles
    Q = np.add.reduce(q, axis=1)

    M = np.sqrt(np.einsum('kd,dd,kd->k', Q, MINKOWSKI, Q))
    b = (-Q[:, 1:] / M[:, np.newaxis])
    x = E_CM / M
    gamma = Q[:, 0] / M
    a = 1. / (1. + gamma)

    bdotq = np.einsum('ki,kpi->kp', b, q[:, :, 1:])

    # make dimensions match
    gamma = gamma[:, np.newaxis]
    x = x[:, np.newaxis]
    p[:, :, 0] = x * (gamma * q[:, :, 0] + bdotq)

    # make dimensions match
    b = b[:, np.newaxis, :]  # dimensions: samples * nparticles * space dim)
    bdotq = bdotq[:, :, np.newaxis]
    x = x[:, :, np.newaxis]
    a = a[:, np.newaxis, np.newaxis]
    p[:, :, 1:] = x * (q[:, :, 1:] + b * q[:, :, 0, np.newaxis] + a * bdotq * b)

    return p.reshape(xs.shape)
