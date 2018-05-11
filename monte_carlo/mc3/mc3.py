"""
Combined Multi channel Markov chain Monte Carlo method to first approximate
and then generate a sample according to a function f that is, in practice,
expensive to evaluate.
"""

import numpy as np

from ..core.densities import Gaussian
from ..core.proposals import UniformLocal
from ..core.markov import MixingMarkovUpdate, DefaultMetropolis
from ..core.integration import MultiChannelMC
from ..hamiltonian import HamiltonianUpdate


# Multi Channel Markov Chain Monte Carlo (combine integral and sampling)
class BasicMC3(object):
    def __init__(self, target, channels, sample_local, beta=.5):
        """ Base implementation of Multi-channel Markov chain Monte Carlo.

        :param target: Function to integrate and sample according to.
        :param channels: Channels object for importance sampling.
        :param sample_local: Markov update (sampler) to explore locally.
        :param beta: Parameter used to decide between update mechanisms.
            The importance sampling Metropolis update is chosen with probability
            beta. Value must be between 0 and 1.
        """
        self.ndim = channels.ndim
        self.target = target
        self.channels = channels
        self.mc_importance = MultiChannelMC(channels)

        self.sample_is = DefaultMetropolis(self.ndim, target, channels)

        self.sample_local = sample_local

        updates = [self.sample_is, self.sample_local]
        self.mixing_sampler = MixingMarkovUpdate(self.ndim, updates)
        self.beta = beta

        self.integration_sample = None

    def integrate(self, *eval_sizes):
        """ Execute multi channel integration and optimization.

        :param eval_sizes: Tuple of three lists of integers, giving the
            sample sizes of each iteration of the three phases of multi channel
            importance sampling (see MonteCarloMultiImportance).
        :return: The integral and error approximates.
        """
        self.integration_sample = self.mc_importance(self.target, *eval_sizes)
        return self.integration_sample

    def sample(self, sample_size, initial=None, **kwargs):
        """ Generate a sample according to the function self.target using MC3.

        Call only after mixing_sampler has been initialized (see __call__).

        :param sample_size: Number of samples to generate.
        :param initial: Initial value in Markov chain.
        :return: Numpy array of shape (sample_size, self.ndim).
        """
        # try to find an initial value
        if initial is None:
            it = 0
            while it < 1000:
                initial = self.sample_is.proposal(None)
                if self.target.pdf(initial) != 0:
                    break
            if self.target.pdf(initial) == 0:
                raise RuntimeError("Could not find a suitable initial value "
                                   "using the multi channel distribution.")

        return self.mixing_sampler.sample(sample_size, initial, **kwargs)

    @property
    def beta(self):
        return self.mixing_sampler.weights[0]

    @beta.setter
    def beta(self, beta):
        self.mixing_sampler.weights = np.array([beta, 1 - beta])

    def __call__(self, eval_sizes, sample_size, initial=None, **kwargs):
        """ Execute MC3 algorithm; integration phase followed by sampling.

        :param eval_sizes: Tuple of three lists of integers,
            giving the sample sizes of each iteration of the three phases of
            multi channel importance sampling (see MonteCarloMultiImportance).
        :param sample_size: Number of samples to generate
        :param initial: Initial value in Markov chain.
        :return: Numpy array of shape (sample_size, self.ndim). The sample
            generated, following the distribution self.target.
        """
        self.integrate(*eval_sizes)

        return self.sample(sample_size, initial, **kwargs)


class MC3Uniform(BasicMC3):

    def __init__(self, fn, channels, delta, beta=.5):
        ndim = channels.ndim
        sample_local = DefaultMetropolis(
            ndim, fn, UniformLocal(ndim, delta))
        super().__init__(fn, channels, sample_local, beta)

    @property
    def delta(self):
        return self.sample_local.delta

    @delta.setter
    def delta(self, delta):
        self.sample_local.delta = delta


class MC3Hamilton(BasicMC3):

    def __init__(self, target_density, channels, m, steps, step_size, beta=.5):

        self.p_dist = Gaussian(channels.ndim, scale=m)
        sample_local = HamiltonianUpdate(
            target_density, self.p_dist, steps, step_size)

        super().__init__(target_density, channels, sample_local, beta)

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

    @property
    def m(self):
        return np.sqrt(self.p_dist.cov)

    @m.setter
    def m(self, value):
        self.p_dist.cov = value ** 2
