"""
Module implements Hamilton Monte Carlo methods for sampling.
"""

import numpy as np
from ..core.markov import MetropolisUpdate


class HamiltonLeapfrog(object):

    def __init__(self, pot_gradient, kin_gradient, step_size, steps):
        """ Leapfrog method to simulate Hamiltonian propagation.

        This method is based on a general structure of the Hamiltonian of
        H = kinetic(p) + potential(q),
        where q is the "space" and p the "momentum" variable.

        :param pot_gradient: Partial derivative of the potential with
            respect to q.
        :param kin_gradient: Partial derivative of the kinetic energy
            with respect to p.
        :param step_size: Size of a simulation step in "time"-space.
        :param steps: Number of iterations to perform in each call.

        """
        self.kin_gradient = kin_gradient
        self.pot_gradient = pot_gradient
        self.step_size = step_size
        self.steps = steps

    def __call__(self, q_init, p_init):
        """ Propagate the state q, p using a given number of simulation steps.

        :param q_init: Initial space variable.
        :param p_init: Initial momentum variable.
        :return: Tuple (q_next, p_next) of state after given number of
            simulation steps.
        """
        p = np.array(p_init, copy=True, ndmin=1, subok=True)
        q = np.array(q_init, copy=True, ndmin=1, subok=True)
        for i in range(self.steps):
            p -= self.step_size / 2 * self.pot_gradient(q)
            q += self.step_size * self.kin_gradient(p)
            p -= self.step_size / 2 * self.pot_gradient(q)
        return q, p


class HamiltonianUpdate(MetropolisUpdate):

    def __init__(self, ndim, m, pot, dpot_dq, simulate):
        super().__init__(ndim)
        self.m = m
        self.simulate = simulate
        self.pot = pot
        self.pot_gradient = dpot_dq

        # artificially introduced state
        self._p_state = None
        self._p_candidate = None

    def accept(self, state, candidate):
        prob = np.exp(-self.pot(candidate) +
                      self.pot(state) -
                      np.sum(self._p_candidate ** 2 / self.m / 2) +
                      np.sum(self._p_state ** 2 / self.m / 2))
        return prob

    def proposal(self, state):
        # first update
        p_state = np.random.normal(0, np.sqrt(self.m), state.size)
        self._p_state = p_state

        # second update
        q, p = self.simulate(state, p_state)
        # negation makes the update reversible, but method is symmetric
        # in p already so practically irrelevant
        # p *= -1
        self._p_candidate = p

        return q

    @property
    def step_size(self):
        return self.simulate.step_size

    @step_size.setter
    def step_size(self, step_size):
        self.simulate.step_size = step_size

    @property
    def steps(self):
        return self.simulate.steps

    @steps.setter
    def steps(self, steps):
        self.simulate.steps = steps


class HMCGaussUpdate(HamiltonianUpdate):
    def __init__(self, ndim, pot, pot_gradient, m,
                 step_size, steps, simulation_method=HamiltonLeapfrog):
        """ Hamilton Monte Carlo Metropolis algorithm.

        The variable of interest is referred to as q, pot is the log
        probability density.

        The momentum variables are artificially introduced and not returned.
        The momenta are sampled according to a Gaussian normal distribution
        with variance m.

        Example:
            For a Gaussian with variance 1 the log probability is q^2/2.
            >>> pot = lambda q: q**2 / 2
            >>> pot_gradient = lambda q: q
            >>> hmcm = HMCGaussUpdate(1, pot, pot_gradient, 1, 10, 1)
            >>> hmcm.init_sampler(0.0)
            >>> # sample 1000 points that will follow a Gaussian
            >>> points = hmcm.sample(1000)

        :param ndim: Dimension of the variable of interest q.
        :param pot: The desired log probability density (of q).
        :param pot_gradient: Partial derivative with respect to q of pot.
        :param m: Variances of the "momentum" distribution.
        :param steps: Number of simulation steps in the update.
        :param step_size: Step size for the simulation method.
        :param simulation_method: Class used to simulate the steps.
            A custom implementation must follow that of HamiltonLeapfrog.
        """
        m = np.array(m, copy=False, subok=True, ndmin=1)
        self.pot = pot
        self.pot_gradient = pot_gradient

        simulate = simulation_method(pot_gradient, self.kin_gradient,
                                     step_size, steps)

        super().__init__(ndim, m, pot, pot_gradient, simulate)

    def kin_gradient(self, p):
        return p / self.m  # Gaussian
