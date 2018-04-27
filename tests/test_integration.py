from monte_carlo.core import *
from monte_carlo import densities

from unittest import TestCase


class TestPlainMC(TestCase):
    def test_linear(self):
        eval_count = 10000
        mc = MonteCarloPlain(2)
        est, err = mc(lambda x, y: x + y, eval_count)
        self.assertAlmostEqual(est, 1.0, 1)  # should match to 1 decimals
        self.assertLess(err, 0.1)  # error shouldn't be too large


class TestImportanceMC(TestCase):
    eval_count = 10000

    def test_plain(self):
        dist = densities.Uniform(2)

        mc = MonteCarloImportance(dist)
        est, err = mc(lambda x, y: x + y, self.eval_count)
        self.assertAlmostEqual(est, 1.0, 1)  # should match to 1 decimals
        self.assertLess(err, 0.1)  # error shouldn't be too large

    def test_sqrt(self):
        # equivalent to plain MC
        dist = densities.make_dist_vect(
            2, lambda x, y: 1 / np.sqrt(x) / np.sqrt(y),
            lambda n: np.sqrt(np.random.random((n, 2))))

        mc = MonteCarloImportance(dist)
        est, err = mc(lambda x, y: x + y, self.eval_count)
        self.assertAlmostEqual(est, 1.0, 0)
        self.assertLess(err, 0.1)  # error shouldn't be too large

    def test_ideal(self):
        dist = densities.make_dist(1, lambda x: 2 * x,
                                   lambda size: np.random.rand(size) ** 2)

        mc_imp = MonteCarloImportance(dist)
        est, err = mc_imp(lambda x: x, 1000)
        # the pdf is ideal since fn(x)/(2*x) = 1/2 = const
        self.assertEqual(0.5, est)
        self.assertEqual(0.0, err)


class TestMultiMC(TestCase):

    def test_plain(self):
        # equivalent to plain monte carlo
        channels = MultiChannel([densities.Uniform(1)])
        mc_imp = MonteCarloMultiImportance(channels)
        est, err = mc_imp(lambda x: x, [], [100], [])
        self.assertAlmostEqual(est, 0.5, 0)
        self.assertLess(err, 0.1)


class TestStratMC(TestCase):

    def test_linear(self):
        # Divide the integration space into 4 equally sized partitions with a
        # base number of 10 sample points in each volume.
        volumes = GridVolumes(ndim=1, divisions=4, default_count=10)
        mc_strat = MonteCarloStratified(volumes=volumes)
        # Stratified sampling expects a multiple instead of a total sample size.
        est, err = mc_strat(lambda x: x, 5)  # 5 * 10 sample points per region
        self.assertAlmostEqual(est, 0.5, 0)
        self.assertLess(err, 0.1)

    def test_interface(self):
        volumes = GridVolumes(ndim=1, divisions=4, default_count=100)
        mc_strat = MonteCarloStratified(volumes=volumes)
        mc_strat = mc_strat.get_interface_infer_multiple()
        est, err = mc_strat(lambda x: x, 4000)  # multiple is 4000/(4*100) = 40
        self.assertAlmostEqual(est, 0.5, 1)
        self.assertLess(err, 0.1)
