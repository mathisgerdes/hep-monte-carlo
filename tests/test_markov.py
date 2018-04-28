from unittest import TestCase

from monte_carlo import *


class MetropTest(TestCase):

    def test_uniform(self):
        pdf = lambda x: np.sin(10 * x) ** 2
        met = make_metropolis(1, pdf)  # 1 dimensional
        met.init_sampler(0.1)  # initialize with start value
        sample = met.sample(1000)  # generate 1000 samples
        self.assertEqual(sample.shape, (1000, 1))

    def test_local(self):
        ndim = 2
        delta = 0.01

        def local(state):
            base = np.maximum(np.zeros(ndim), state - delta / 2)
            base_capped = np.minimum(base, np.ones(ndim) - delta)
            return base_capped + np.random.rand() * delta

        dist = make_dist_vect(ndim, lambda x, y: np.sin(10 * x * y) ** 2)
        met = make_metropolis(ndim, dist.pdf, local)  # 1 dimensional
        met.init_sampler([0.1, 0.1])  # initialize with start value
        sample = met.sample(1000)  # generate 1000 samples
        self.assertEqual(sample.shape, (1000, 2))
