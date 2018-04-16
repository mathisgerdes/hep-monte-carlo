"""
Combined Multi channel Markov chain Monte Carlo method to first approximate
and then generate a sample according to a function f that is, in practice,
expensive to evaluate.
"""

import numpy as np

from monte_carlo.markov import MetropolisUpdate, MixingMetropolisUpdate
from monte_carlo.integration import MonteCarloMultiImportance
from monte_carlo.hmc import HMCGaussUpdate


# Multi Channel Markov Chain Monte Carlo (combine integral and sampling)
class AbstractMC3(object):
    def __init__(self, dim, fn, channels, sample_local, beta=.5):
        """

        :param dim: Dimensionality of sample space.
        :param fn: Function to integrate and sample according to.
        :param channels: Channels object for importance sampling.
        :param sample_local: Markov update (sampler) to explore locally.
        :param beta: Parameter used to decide between update mechanisms.
            The importance sampling Metropolis update is chosen with probability
            beta. Value must be between 0 and 1.
        """
        self.dim = dim
        self.fn = fn
        self.channels = channels
        self.mc_importance = MonteCarloMultiImportance(channels)

        self.sample_is = MetropolisUpdate(
            dim, self.fn,
            proposal=lambda _: self.channels.sample(1)[0],
            proposal_pdf=lambda _, c: self.channels.pdf(c))

        self.sample_local = sample_local

        updates = [self.sample_is, self.sample_local]
        self.mixing_sampler = MixingMetropolisUpdate(dim, updates)
        self.beta = beta

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

    def sample(self, sample_size):
        """ Generate a sample according to the function self.fn using MC3.

        Call only after mixing_sampler has been initialized (see __call__).

        :param sample_size: Number of samples to generate.
        :return: Numpy array of shape (sample_size, self.dim).
        """
        return self.mixing_sampler.sample(sample_size)

    @property
    def beta(self):
        return self.mixing_sampler.weights[0]

    @beta.setter
    def beta(self, beta):
        self.mixing_sampler.weights = np.array([beta, 1 - beta])

    def __call__(self, integration_sample_sizes, sample_size, initial=None):
        """ Execute MC3 algorithm; integration phase followed by sampling.

        :param integration_sample_sizes: Tuple of three lists of integers,
            giving the sample sizes of each iteration of the three phases of
            multi channel importance sampling (see MonteCarloMultiImportance).
        :param sample_size: Number of samples to generate
        :return: Numpy array of shape (sample_size, self.dim). The sample
            generated, following the distribution self.fn.
        """
        self.integrate(*integration_sample_sizes)

        # try to find an initial value
        if initial is None:
            it = 0
            while it < 1000:
                initial = self.sample_is.proposal(None)
                if self.fn(initial) != 0:
                    break
            if self.fn(initial) == 0:
                raise RuntimeError("Could not find a suitable initial value"
                                   "using the multi channel distribution.")

        self.mixing_sampler.init_sampler(initial)
        return self.sample(sample_size)


class MC3Uniform(AbstractMC3):

    def __init__(self, dim, fn, channels, delta, beta=.5):

        sample_local = MetropolisUpdate(dim, fn, self.generate_local)
        super().__init__(dim, fn, channels, sample_local, beta)

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

    def __init__(self, dim, fn, channels, dpot_dq, m, step_size, steps,
                 beta=.5):

        sample_local = HMCGaussUpdate(dim, lambda *x: -np.log(fn(*x)),
                                      dpot_dq, m, step_size, steps)

        super().__init__(dim, fn, channels, sample_local, beta)

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
