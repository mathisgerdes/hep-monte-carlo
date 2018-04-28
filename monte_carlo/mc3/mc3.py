"""
Combined Multi channel Markov chain Monte Carlo method to first approximate
and then generate a sample according to a function f that is, in practice,
expensive to evaluate.
"""

import numpy as np

from ..core.densities import Gaussian
from ..core.markov import make_metropolis, MixingMarkovUpdate
from ..core.integration import MonteCarloMultiImportance
from ..hamiltonian import HamiltonianUpdate


# Multi Channel Markov Chain Monte Carlo (combine integral and sampling)
class AbstractMC3(object):
    def __init__(self, fn, channels, sample_local, beta=.5):
        """ Base implementation of Multi-channel Markov chain Monte Carlo.

        :param ndim: Dimensionality of sample space.
        :param fn: Function to integrate and sample according to.
        :param channels: Channels object for importance sampling.
        :param sample_local: Markov update (sampler) to explore locally.
        :param beta: Parameter used to decide between update mechanisms.
            The importance sampling Metropolis update is chosen with probability
            beta. Value must be between 0 and 1.
        """
        self.ndim = channels.ndim
        self.fn = fn
        self.channels = channels
        self.mc_importance = MonteCarloMultiImportance(channels)

        self.sample_is = make_metropolis(
            self.ndim, self.fn, self.channels.proposal, self.channels.pdf)

        self.sample_local = sample_local

        updates = [self.sample_is, self.sample_local]
        self.mixing_sampler = MixingMarkovUpdate(self.ndim, updates)
        self.beta = beta

        self.int_est, self.int_err = None, None  # store integration results

    def integrate(self, *eval_sizes):
        """ Execute multi channel integration and optimization.

        :param eval_sizes: Tuple of three lists of integers, giving the
            sample sizes of each iteration of the three phases of multi channel
            importance sampling (see MonteCarloMultiImportance).
        :return: The integral and error approximates.
        """
        self.int_est, self.int_err = self.mc_importance(self.fn, *eval_sizes)
        return self.int_est, self.int_err

    def sample(self, sample_size, initial=None, **kwargs):
        """ Generate a sample according to the function self.fn using MC3.

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
                if self.fn(initial) != 0:
                    break
            if self.fn(*initial) == 0:
                raise RuntimeError("Could not find a suitable initial value "
                                   "using the multi channel distribution.")

        self.mixing_sampler.init_sampler(initial, **kwargs)

        return self.mixing_sampler.sample(sample_size)

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
            generated, following the distribution self.fn.
        """
        self.integrate(*eval_sizes)

        return self.sample(sample_size, initial, **kwargs)


class MC3Uniform(AbstractMC3):

    def __init__(self, fn, channels, delta, beta=.5):

        sample_local = make_metropolis(channels.ndim, fn, self.generate_local)
        super().__init__(fn, channels, sample_local, beta)

        if np.ndim(delta) == 0:
            delta = np.ones(self.ndim) * delta
        elif len(delta) == self.ndim:
            delta = np.array(delta)
        else:
            raise ValueError("delta must be a float or an array of "
                             "length ndim.")
        self.delta = delta

    def generate_local(self, state):
        """ Uniformly generate a point in a self.delta environment.

        If state is close to the edges in a dimension, the uniform range
        is asymmetric such that the length is preserved but within the unit
        hypercube [0, 1]^ndim.

        :param state: The state (point) around which to sample.
        :return: Numpy array of length self.ndim
        """
        base = np.maximum(np.zeros(self.ndim), state - self.delta / 2)
        base_capped = np.minimum(base, np.ones(self.ndim) - self.delta)
        return base_capped + np.random.rand() * self.delta


class MC3Hamilton(AbstractMC3):

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
