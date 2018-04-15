"""
Combined Multi channel Markov chain Monte Carlo method to first approximate
and then generate a sample according to a function f that is, in practice,
expensive to evaluate.
"""

import numpy as np

from monte_carlo.markov import MetropolisSampler, MixingMetropolisSampler
from monte_carlo.integration import MonteCarloMultiImportance
from monte_carlo.hmc import HMCMetropolisGauss


class AbstractMC3(object):
    def __init__(self, dim, channels, fn, initial_value, sample_local):
        self.channels = channels
        self.mc_importance = MonteCarloMultiImportance(channels)

        self.fn = fn
        self.dim = dim

        self.sample_is = MetropolisSampler(
            initial_value, self.fn,  # initial value will be ignored later
            proposal_pdf=lambda s, c: self.channels.pdf(c),
            proposal=lambda s: self.channels.sample(1)[0])

        self.sample_local = sample_local

        updates = [self.sample_is, self.sample_local]
        self.sampler = MixingMetropolisSampler(initial_value, updates)

        self.int_est, self.int_err = None, None  # store integration results

    def integrate(self, *sample_sizes):
        """ Execute multi channel integration and optimization.

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
        self.sampler.weights = np.array([beta, 1-beta])
        return self.sampler(sample_size)

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
        self.integrate(*integration_sample_sizes)
        return self.sample(sample_size, beta)


# Multi Channel Markov Chain Monte Carlo (combine integral and sampling)
class MC3Uniform(AbstractMC3):

    def __init__(self, dim, channels, fn, delta, initial_value):

        sample_local = MetropolisSampler(
            initial_value, fn, proposal=self.generate_local)
        AbstractMC3.__init__(
            self, dim, channels, fn, initial_value, sample_local)

        if np.ndim(delta) == 0:
            delta = np.ones(dim) * delta
        elif len(delta) == dim:
            delta = np.array(delta)
        else:
            raise ValueError("delta must a float or an array of length dim.")
        self.delta = delta

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


class MC3Hamilton(AbstractMC3):

    def __init__(self, dim, channels, fn, dpot_dq, m, initial_value,
                 step_size, steps):

        sample_local = HMCMetropolisGauss(
            initial_value, lambda *x: -np.log(fn(*x)),
            dpot_dq, m, step_size, steps)

        AbstractMC3.__init__(
            self, dim, channels, fn, initial_value, sample_local)

    @property
    def step_size(self):
        return self.sample_local.step_size

    @step_size.setter
    def step_size(self, step_size):
        self.sample_local.step_size = step_size

    @property
    def steps(self):
        return self.sample_local.steps

    @steps.setter
    def steps(self, steps):
        self.sample_local.steps = steps
