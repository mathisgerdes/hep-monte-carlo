import numpy as np
from .. import *

from unittest import TestCase


class TestPlainMC(TestCase):
    def test_linear(self):
        eval_count = 10000
        mc = PlainMC(2)
        sample = mc(lambda x, y: x + y, eval_count)
        # should match to 1 decimals
        self.assertAlmostEqual(sample.integral, 1.0, 1)
        # error shouldn't be too large
        self.assertLess(sample.integral_err, 0.1)


class TestImportanceMC(TestCase):
    eval_count = 10000

    def test_plain(self):
        dist = densities.Uniform(2)

        mc = ImportanceMC(dist)
        sample = mc(lambda x, y: x + y, self.eval_count)
        # should match to 1 decimals
        self.assertAlmostEqual(sample.integral, 1.0, 1)
        # error shouldn't be too large
        self.assertLess(sample.integral_err, 0.1)

    def test_sqrt(self):
        # equivalent to plain MC
        dist = Distribution.make(
            pdf_vect=lambda x, y: 1 / np.sqrt(x) / np.sqrt(y), ndim=2,
            rvs=lambda n: np.sqrt(np.random.random((n, 2))))

        mc = ImportanceMC(dist)
        sample = mc(lambda x, y: x + y, self.eval_count)
        # should match to 1 decimals
        self.assertAlmostEqual(1.0, sample.integral, 0)
        # error shouldn't be too large
        self.assertLess(sample.integral_err, 0.1)

    def test_ideal(self):
        dist = Distribution.make(
            lambda x: 2 * x, ndim=1, rvs=lambda size: np.random.rand(size) ** 2)

        mc_imp = ImportanceMC(dist)
        sample = mc_imp(lambda x: x, 1000)
        # the pdf is ideal since fn(x)/(2*x) = 1/2 = const
        self.assertEqual(0.5, sample.integral)
        self.assertEqual(0.0, sample.integral_err)


class TestMultiMC(TestCase):

    def test_plain(self):
        # equivalent to plain monte carlo
        channels = MultiChannel([densities.Uniform(1)])
        mc_imp = MultiChannelMC(channels)
        sample = mc_imp(lambda x: x, [], [100], [])
        self.assertAlmostEqual(0.5, sample.integral, 0)
        self.assertLess(sample.integral_err, 0.1)


class TestStratMC(TestCase):

    def test_linear(self):
        # Divide the integration space into 4 equally sized partitions with a
        # base number of 10 sample points in each volume.
        volumes = GridVolumes(ndim=1, divisions=4, default_base_count=10)
        mc_strat = StratifiedMC(volumes=volumes)
        # Stratified sampling expects a multiple instead of a total sample size.
        sample = mc_strat(lambda x: x, 5)  # 5 * 10 sample points per region
        self.assertAlmostEqual(sample.integral, 0.5, 0)
        self.assertLess(sample.integral_err, 0.1)

    def test_interface(self):
        volumes = GridVolumes(ndim=1, divisions=4, default_base_count=100)
        mc_strat = StratifiedMC(volumes=volumes)
        mc_strat = mc_strat.get_interface_infer_multiple()
        sample = mc_strat(lambda x: x, 4000)  # multiple is 4000/(4*100) = 40
        self.assertAlmostEqual(sample.integral, 0.5, 1)
        self.assertLess(sample.integral_err, 0.1)
