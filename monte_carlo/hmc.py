"""
Module implements Hamilton Monte Carlo methods for sampling.
"""

import numpy as np
from monte_carlo.sampling import GenericMetropolis


class HamiltonLeapfrog(object):

    def __init__(self, dpot_dq, dkin_dp, step_size, steps):
        """ Leapfrog method to simulate Hamiltonian propagation.

        This method is based on a general structure of the Hamiltonian of
        H = kinetic(p) + potential(q),
        where q is the "space" and p the "momentum" variable.

        :param dpot_dq: Partial derivative of the potential with respect to q.
        :param dkin_dp: Partial derivative of the kinetic energy
            with respect to p.
        :param step_size: Size of a simulation step in "time"-space.
        :param steps: Number of iterations to perform in each call.

        """
        self.dkin_dp = dkin_dp
        self.dpot_dq = dpot_dq
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
            p -= self.step_size / 2 * self.dpot_dq(q)
            q += self.step_size * self.dkin_dp(p)
            p -= self.step_size / 2 * self.dpot_dq(q)
        return q, p


class HMCMetropolisGauss(GenericMetropolis):
    def __init__(self, initial_q, dim_q, pot, dpot_dq, m,
                 steps, step_size, simulation_method=HamiltonLeapfrog):
        """ Hamilton Monte Carlo Metropolis algorithm.

        The variable of interest is referred to as q, pot is the log
        probability density.

        The momentum variables are artificially introduced, such that the total
        dimensionality of the state in the Metropolis algorithm is 2*dim_q.
        The momenta are sampled according to a Gaussian normal distribution
        with variance m.

        The default call method returns only the "q" part of the states
        (i.e. the variables of interest). To get the values of the "momenta"
        use full_sample.

        Example:
            For a Gaussian with variance 1 the log probability is q^2/2.
            >>> pot = lambda q: q**2 / 2
            >>> dpot_dq = lambda q: q
            >>> hmcm = HMCMetropolisGauss(0.0, 1, pot, dpot_dq, 1, 10, 1)
            >>> # sample 1000 points that will follow a Gaussian
            >>> points = hmcm(1000)

        :param initial_q: Initial value of the variable of interest q.
        :param dim_q: Dimension of the variable of interest q.
        :param pot: The desired log probability density (of q).
        :param dpot_dq: Partial derivative with respect to q of pot.
        :param m: Variances of the "momentum" distribution.
        :param steps: Number of simulation steps in the update.
        :param step_size: Step size for the simulation method.
        :param simulation_method: Class used to simulate the steps.
            A custom implementation must follow that of HamiltonLeapfrog.
        """
        self.m = np.array(m, copy=False, subok=True, ndmin=1)
        self.pot = pot
        self.dpot_dq = dpot_dq
        self.dim_q = dim_q

        self.simulation_method = simulation_method(dpot_dq, self.dkin_dp,
                                                   step_size, steps)

        initial = np.empty(2 * dim_q)
        initial[:dim_q] = initial_q
        initial[dim_q:] = np.random.normal(0, np.sqrt(self.m), dim_q)
        super().__init__(initial)

    def dkin_dp(self, p):
        return p / self.m  # Gaussian

    def accept(self, state, candidate):
        q, p = state[:self.dim_q], state[self.dim_q:]
        q1, p1 = candidate[:self.dim_q], candidate[self.dim_q:]
        prob = np.exp(-self.pot(q1) +
                      self.pot(q) -
                      np.sum(p1 ** 2 / self.m / 2) +
                      np.sum(p ** 2 / self.m / 2))
        return prob

    def proposal(self, state):
        q, p = state[:self.dim_q], state[self.dim_q:]

        # first update step
        p = np.random.normal(0, np.sqrt(self.m), self.m.size)

        # second update step
        candidate = np.empty(self.dim)
        candidate[:self.dim_q], candidate[self.dim_q:] = \
            self.simulation_method(q, p)
        candidate[self.dim_q:] *= -1  # negation makes the update reversible

        return candidate

    def full_sample(self, sample_size, get_accept_rate):
        return super().__call__(sample_size, get_accept_rate)

    def __call__(self, sample_size=1, get_accept_rate=False):
        res = super().__call__(sample_size, get_accept_rate=get_accept_rate)
        if get_accept_rate:
            return res[0][:, :self.dim_q], res[1]

        return res[:, :self.dim_q]
