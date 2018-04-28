"""
Module implements Hamilton Monte Carlo methods for sampling.
"""

from ..core.markov import MetropolisUpdate, StateArray
from .simulation import HamiltonLeapfrog


class HamiltonianUpdate(MetropolisUpdate):

    def __init__(self, target_density, p_dist,
                 steps, step_size, sim_method=HamiltonLeapfrog):
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

        :param pot: The desired log probability density (of q).
        :param pot_gradient: Partial derivative with respect to q of pot.
        :param m: Variances of the "momentum" distribution.
        :param steps: Number of simulation steps in the update.
        :param step_size: Step size for the simulation method.
        :param simulation_method: Class used to simulate the steps.
            A custom implementation must follow that of HamiltonLeapfrog.
        """
        super().__init__(target_density.ndim, target_density.pdf)
        self.is_hasting = False
        self.p_dist = p_dist
        self.target_density = target_density
        self.simulate = sim_method(
            target_density.pot_gradient, p_dist.pot_gradient, step_size, steps)

    def accept(self, state, candidate):
        prob = (candidate.pdf * self.p_dist.pdf(candidate.momentum) /
                state.pdf / self.p_dist.pdf(state.momentum))
        return prob

    def proposal_pdf(self, state, candidate):
        pass  # update is Metropolis-like

    def proposal(self, state):
        # first update
        state.momentum = self.p_dist.proposal()

        # second update
        q, p = self.simulate(state, state.momentum)
        # negation makes the update reversible, but method is symmetric
        # in p already so practically irrelevant
        # p *= -1

        if q is None:
            pdf = 0
        else:
            pdf = self.target_density.pdf(q)
        return StateArray(q, momentum=p, pdf=pdf)

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
