import numpy as np
from .base import MetropolisUpdate, StateArray


class AdaptiveMetropolisUpdate(MetropolisUpdate):
    """
    adaptive Metropolis-Hastings sampler according to Haario (2001)
    """

    def __init__(self, ndim, target_pdf, proposal_dist,
                 t_initial, adapt_schedule):
        super().__init__(ndim, target_pdf, is_adaptive=True)

        self.proposal_dist = proposal_dist

        self.t_initial = t_initial
        self.adapt_schedule = adapt_schedule
        self.s_d = 2.4 ** 2 / self.ndim
        self.epsilon = 1e-6
        
        self.mean = None
        self.mean_previous = None
        self.cov = self.proposal_dist.cov

    def proposal(self, state):
        prop = self.proposal_dist.proposal()
        return StateArray(prop, pdf=self.pdf(prop))

    def proposal_pdf(self, state, candidate):
        return self.proposal_dist(candidate)
    
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
                self.proposal_dist.cov = self.cov

# class RobustAdaptiveMetropolis(StaticMetropolis):
#    """
#    rebustadaptive Metropolis-Hastings sampler according to Vihola (2012)
#    """
