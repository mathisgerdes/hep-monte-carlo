"""
Module implements Hamilton Monte Carlo methods for sampling.
"""
import numpy as np
from ..core.markov.metropolis import MetropolisUpdate, MetropolisState
from .simulation import HamiltonLeapfrog


class HamiltonState(MetropolisState):
    def __new__(cls, input_array, momentum=None, **kwargs):
        obj = super().__new__(cls, input_array, **kwargs)
        if momentum is not None:
            obj.momentum = momentum
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return  # was called from the __new__ above
        super().__array_finalize__(obj)
        self.momentum = getattr(obj, 'momentum', None)


class HamiltonianUpdate(MetropolisUpdate):

    def __init__(self, target_density, p_dist,
                 steps, step_size, simulate=None, sim_class=HamiltonLeapfrog,
                 is_adaptive=False):
        """ Hamilton Monte Carlo Metropolis algorithm.

        The variable of interest is referred to as q, pot is the log
        probability density.

        The momentum variables are artificially introduced and not returned.
        The momenta are sampled according to a Gaussian normal distribution
        with variance m.

        :param target_density: The desired density (of q).
        :param steps: Number of simulation steps in the update.
        :param step_size: Step size for the simulation method.
        :param sim_method: Class used to simulate the steps.
            A custom implementation must follow that of HamiltonLeapfrog.
        """
        super().__init__(target_density.ndim, target_density.pdf, is_adaptive)
        self.p_dist = p_dist
        self.target_density = target_density

        if simulate is None:
            self.simulate = sim_class(
                target_density.pot_gradient,
                p_dist.pot_gradient, step_size, steps)
        else:
            self.simulate = simulate

    def accept(self, state, candidate):
        try:
            prob = (candidate.pdf * self.p_dist.pdf(candidate.momentum) /
                    state.pdf / self.p_dist.pdf(state.momentum))
            if np.isnan(prob):
                return 0
            return prob
        except RuntimeWarning:
            return 0

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
        return HamiltonState(q, momentum=p, pdf=pdf)

    def init_state(self, state):
        if not isinstance(state, HamiltonState):
            state = HamiltonState(state)
        if state.momentum is None:
            state.momentum = self.p_dist.proposal()
        return super().init_state(state)

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

    def sample(self, sample_size, initial, out_mask=None, log_every=5000):
        sample = super().sample(sample_size, initial, out_mask, log_every)
        sample.target = self.target_density
        return sample
