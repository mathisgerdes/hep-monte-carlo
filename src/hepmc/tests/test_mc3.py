from unittest import TestCase

from .. import *
from ..mc3 import *


class PlainMC3Test(TestCase):

    def test_basic(self):
        count = 3000

        def fn(x):
            return np.sin(10 * x)
        channels = MultiChannel([densities.Uniform(1)])
        mc3_sampler = MC3Uniform(Density.make(fn, 1),
                                 channels, delta=.01, beta=1)

        res = mc3_sampler(([], [500] * 40, []), count, initial=.5, log_every=0)
        self.assertTrue(mc3_sampler.sample_is.is_hasting)
        self.assertEqual(res.data.shape, (count, 1))


class HamiltonMC3Test(TestCase):

    def test_basic(self):
        count = 3000

        def fn(x):
            return np.sin(10 * x)

        def dfn(x):
            return 10 * np.cos(10 * x)

        channels = MultiChannel([densities.Uniform(1)])
        pdf = Density.make(fn, ndim=1, pdf_gradient=dfn)
        mc3_sampler = MC3Hamilton(pdf, channels, np.ones(1),
                                  steps=10, step_size=.1, beta=1)

        res = mc3_sampler(([], [500] * 40, []), count, .5, log_every=0)
        self.assertEqual(res.data.shape, (count, 1))
