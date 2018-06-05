import numpy as np
from .. import *

from unittest import TestCase


class MetropTest(TestCase):

    def test_mean(self):
        met = DefaultMetropolis(1, lambda x: 1)
        sample = met.sample(1000, 0.1)  # generate 1000 samples
        self.assertAlmostEqual(0.5, sample.mean[0], 1)

    def test_uniform(self):
        pdf = lambda x: np.sin(10 * x) ** 2
        met = DefaultMetropolis(1, pdf)  # 1 dimensional
        sample = met.sample(1000, .1)  # generate 1000 samples
        self.assertEqual(sample.data.shape, (1000, 1))

    def test_local(self):
        ndim = 2
        delta = 0.01

        local = proposals.UniformLocal(ndim, delta)

        dist = Density.make(
            pdf_vect=lambda x, y: np.sin(10 * x * y) ** 2, ndim=ndim)
        met = DefaultMetropolis(ndim, dist.pdf, local)  # 1 dimensional
        sample = met.sample(1000, [0.1, 0.1])  # generate 1000 samples
        self.assertEqual(sample.data.shape, (1000, 2))


class AdaptiveMetTest(TestCase):
    def test_mean(self):
        proposal_dist = densities.Gaussian(1)
        met = AdaptiveMetropolisUpdate(
            1, lambda x: (0 < x) * (x < 1), proposal_dist, 10, lambda t: t<100)
        sample = met.sample(1000, .1)  # generate 1000 samples
        self.assertAlmostEqual(0.5, sample.mean[0], 0)


class HamiltonTest(TestCase):

    def test_gauss(self):
        s = 1
        # hamilton monte carlo
        density = densities.Gaussian(1)
        momentum_dist = densities.Gaussian(1, scale=1)
        hmcm = hamiltonian.HamiltonianUpdate(
            density, momentum_dist, steps=10, step_size=1)
        res = hmcm.sample(100, 0.)
        self.assertEqual((100, 1), res.data.shape)


class HamiltonDualAvTest(TestCase):

    def test_gauss(self):
        s = 1
        dh_dq = lambda q: q
        pdf = lambda x: np.exp(-x ** 2 / 2 / s ** 2) / np.sqrt(
            2 * np.pi * s ** 2)
        # hamilton monte carlo
        density = Density.make(pdf, ndim=1, pdf_gradient=dh_dq)
        momentum_dist = densities.Gaussian(1, scale=1)
        hmcm = hamiltonian.DualAveragingHMC(
            density, momentum_dist, 1, lambda t: t < 10)
        res = hmcm.sample(100, 0.)
        self.assertEqual((100, 1), res.data.shape)


class HamiltonNUTSTest(TestCase):

    def test_gauss(self):
        s = 1
        dh_dq = lambda q: q
        pdf = lambda x: np.exp(-x ** 2 / 2 / s ** 2) / np.sqrt(
            2 * np.pi * s ** 2)
        # hamilton monte carlo
        density = Density.make(pdf, ndim=1, pdf_gradient=dh_dq)
        momentum_dist = densities.Gaussian(1, scale=1)
        hmcm = hamiltonian.NUTSUpdate(density, momentum_dist, lambda t: t < 10)
        res = hmcm.sample(100, 0.)
        self.assertEqual((100, 1), res.data.shape)


class StaticSphericalHMCTest(TestCase):
    def test_gauss(self):
        s = 1
        dh_dq = lambda q: q
        pdf = lambda x: np.exp(-x ** 2 / 2 / s ** 2) / np.sqrt(
            2 * np.pi * s ** 2)
        # hamilton monte carlo
        density = Density.make(pdf, ndim=1, pdf_gradient=dh_dq)
        hmcm = hamiltonian.StaticSphericalHMC(density, .5, 2, 1, 10)
        res = hmcm.sample(100, 0.1)
        self.assertEqual((100, 1), res.data.shape)

    def test_mean(self):
        density = densities.Uniform(2)
        hmcm = hamiltonian.StaticSphericalHMC(density, .5, 2, 1, 10)
        sample = hmcm.sample(1000, [0.1, 0.1])  # generate 1000 samples
        self.assertAlmostEqual(0.5, sample.mean[0], 0)

    def test_gauss2(self):
        # spherical hmc
        density = densities.Gaussian(1, mu=.6, scale=.1)

        hmcm = hamiltonian.StaticSphericalHMC(density, .9, 1, 10, 100)
        sample = hmcm.sample(100, 0.6)
        self.assertEqual((100, 1), sample.data.shape)


class DualSphericalHMCTest(TestCase):

    def test_mean(self):
        density = densities.Uniform(2)
        hmcm = hamiltonian.DualAveragingSphericalHMC(
            density, 1., lambda t: t < 100)
        sample = hmcm.sample(1000, [0.1, 0.1])  # generate 1000 samples
        self.assertAlmostEqual(0.5, sample.mean[0], 0)
