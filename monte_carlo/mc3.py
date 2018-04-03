"""
Combined Multi channel Markov chain Monte Carlo method to first approximate
and then generate a sample according to a function f that is, in practice,
expensive to evaluate.
"""

import numpy as np

from monte_carlo.sampling import Metropolis, MetropolisHasting
from monte_carlo.integration import MonteCarloMultiImportance


# Multi Channel Markov Chain Monte Carlo (combine integral and sampling)
class MC3(object):

    def __init__(self, dim, channels, fn, delta=None, initial_value=None):
        self.channels = channels
        self.mc_importance = MonteCarloMultiImportance(channels)
        self.fn = fn
        self.dim = dim
        if initial_value is None:
            initial_value = np.random.rand(dim)
        self.sample_IS = MetropolisHasting(initial_value, self.fn,
                                           lambda s, c: self.channels.pdf(c),
                                           lambda s: self.channels.sample(1)[0])
        self.sample_METROPOLIS = Metropolis(initial_value, self.fn,
                                            self.generate_local)

        if np.ndim(delta) == 0:
            delta = np.ones(dim) * delta
        elif delta is None:
            delta = np.ones(dim) * .05
        elif len(delta) == dim:
            delta = np.array(delta)
        else:
            raise ValueError(
                "delta must be None, a float, or an array of length dim.")
        self.delta = delta

        self.int_est, self.int_err = None, None  # store integration results

    def generate_local(self, state):
        """ Uniformly generate a point in a self.delta environment.

        If state is close to the edges in a dimension, the uniform range
        is asymmetric such that the length is preserved but within the unit
        hypercube [0, 1]^dim.

        :param state: The state (point) around which to sample.
        :return: Numpy array of length self.dim
        """
        base = np.maximum(np.zeros(self.dim), state - self.delta / 2)
        base_capped = np.minimum(base, np.ones(self.dim) - self.delta)
        return base_capped + np.random.rand() * self.delta

    def integrate(self, sample_sizes):
        """

        :param sample_sizes: Tuple of three lists of integers, giving the
            sample sizes of each iteration of the three phases of multi channel
            importance sampling (see MonteCarloMultiImportance).
        :return: The integral and error approximates.
        """
        self.int_est, self.int_err = self.mc_importance(self.fn, *sample_sizes)
        return self.int_est, self.int_err

    def sample(self, sample_size, beta):
        """ Generate a sample according to the function self.fn using MC3.

        :param sample_size: Number of samples to generate.
        :param beta: Parameter used to decide between update mechanisms.
            The importance sampling Metropolis update is chosen with probability
            beta. Value must be between 0 and 1.
        :return: Numpy array of shape (sample_size, self.dim).
        """
        sample = np.empty((sample_size, self.dim))

        for i in range(sample_size):
            if np.random.rand() <= beta:
                self.sample_METROPOLIS.state = sample[i] = self.sample_IS(1)
            else:
                self.sample_IS.state = sample[i] = self.sample_METROPOLIS(1)

        return sample

    def __call__(self, integration_sample_sizes, sample_size, beta):
        """ Execute MC3 algorithm; integration phase followed by sampling.

        :param integration_sample_sizes: Tuple of three lists of integers,
            giving the sample sizes of each iteration of the three phases of
            multi channel importance sampling (see MonteCarloMultiImportance).
        :param sample_size: Number of samples to generate
        :param beta: Parameter used to decide between update mechanisms.
            The importance sampling Metropolis update is chosen with probability
            beta. Value must be between 0 and 1.
        :return: Numpy array of shape (sample_size, self.dim). The sample
            generated, following the distribution self.fn.
        """
        self.integrate(integration_sample_sizes)
        return self.sample(sample_size, beta)
