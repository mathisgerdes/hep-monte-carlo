import numpy as np
from .metropolis import DefaultMetropolis


class StochasticOptimizeUpdate(DefaultMetropolis):
    """
    Adapts the cov of a local proposal distribution by dual averaging
    """
    def __init__(self, target_density, local_dist, target_rate=0.6,
                 adapt_schedule=None, kappa=0.75, mult=0.005, t0=1):
        super().__init__(target_density.ndim, target_density, adaptive=True,
                         proposal=local_dist)

        if adapt_schedule is None:
            def adapt_schedule(t):
                return mult * np.power(t, -kappa)

        self.adapt_schedule = adapt_schedule
        self.local_dist = local_dist
        self.target_rate = target_rate
        self.t0 = t0

        # used / set later
        self.accepted = 0
        self.generated = 0

    def init_adapt(self, initial_state):
        self.accepted = 0
        self.generated = 0

    def adapt(self, t, prev, current, accept):
        if t > self.t0:
            Ht = self.target_rate - self.accepted / self.generated
            self.local_dist.cov = np.abs(
                self.local_dist.cov - Ht * self.adapt_schedule(t))

    def next_state(self, state, iteration):
        next_state = super().next_state(state, iteration)
        self.generated += 1
        self.accepted += not np.array_equal(state, next_state)
        return next_state


class AdaptiveMetropolisUpdate(DefaultMetropolis):
    """
    adaptive Metropolis-Hastings sampler according to Haario (2001)
    """

    def __init__(self, ndim, target, proposal,
                 t_initial, adapt_schedule):
        super().__init__(ndim, target, proposal)
        self.is_adaptive = True

        try:
            _ = proposal.cov
        except AttributeError:
            raise RuntimeError("Adaptive Metropolis only works with proposals "
                               "that expose a 'cov' attribute.")

        self.t_initial = t_initial
        self.adapt_schedule = adapt_schedule
        self.s_d = 2.4 ** 2 / self.ndim
        self.epsilon = 1e-6
        
        self.mean = None
        self.mean_previous = None
        self.cov = self._proposal.cov
    
    def adapt(self, t, prev, current, accept):
        if not type(current) is np.ndarray:
            current = np.array(current)
        if t == 1:
            self.mean = current
        else:
            self.mean_previous = self.mean
            self.mean = 1/(t+1) * (t * self.mean + current)
        
        if t > self.t_initial:
            self.cov = (t-1)/t * self.cov + self.s_d/t * (
                    t*np.outer(self.mean_previous, self.mean_previous) -
                    (t+1)*np.outer(self.mean, self.mean) +
                    np.outer(current, current) +
                    self.epsilon*np.identity(self.ndim))
        
            if self.adapt_schedule(t) is True:
                self._proposal.cov = self.cov


# class RobustAdaptiveMetropolis(StaticMetropolis):
#    """
#    rebustadaptive Metropolis-Hastings sampler according to Vihola (2012)
#    """
