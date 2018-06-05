import numpy as np
from .. import densities

from unittest import TestCase


class DensityTest(object):
    class_to_test = None

    def test_pdf(self):
        count = 100
        ndim = 3
        density = self.class_to_test(ndim)
        xs = np.random.random((count, ndim))
        prob = density.pdf(xs)
        self.assertEqual(prob.shape, (count,))
        self.assertTrue(np.all(prob == density(*xs.transpose())))

    def test_pdf_1d(self):
        density = self.class_to_test(1)
        self.assertEqual(density(0, 1, 2).shape, (3,))
        self.assertEqual(density.pdf(0).shape, (1,))


class PotDensityTest(DensityTest):

    def test_pot(self):
        count = 100
        ndim = 3
        density = self.class_to_test(ndim)
        xs = np.random.random((count, ndim))
        pot_grad = density.pot_gradient(xs)
        self.assertEqual((count, ndim), pot_grad.shape)


class DistributionTest(object):
    class_to_test = None

    def test_call(self):
        count = 100
        ndim = 3
        distr = self.class_to_test(ndim)
        res = distr.rvs(count)
        self.assertEqual((count, ndim), res.shape)


class GaussTest(TestCase, DistributionTest, PotDensityTest):
    class_to_test = densities.Gaussian

    def test_call_nd(self):
        count = 100
        ndim = 3
        distr = self.class_to_test(ndim, mu=[1, 10, 0], scale=[1, 1, 1])
        res = distr.rvs(count)
        self.assertEqual(res.shape, (count, ndim))


class UniformTest(TestCase, DistributionTest, PotDensityTest):
    class_to_test = densities.Uniform


class CamelTest(TestCase, PotDensityTest):
    class_to_test = densities.Camel

