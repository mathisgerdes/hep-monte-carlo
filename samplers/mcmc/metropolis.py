from samplers.mcmc.mcmc_base import MCMC
import numpy as np

class StaticMetropolis(MCMC):
    """
    basic Metropolis-Hastings sampler
    """
    
    def __init__(self, ndim, target_pdf, proposal_dist, is_adaptive=False):
        super().__init__(ndim, target_pdf, proposal_dist, is_adaptive)
        
    def proposal(self, current, current_pdf):
        proposal = self.proposal_dist.rvs(current)
        proposal_pdf = self.target_pdf(proposal)
        
        return proposal, proposal_pdf
    
    def sample(self, nsamples, start):
        samples = [start]
        current_pdf = self.target_pdf(start)
        for t in range(1, nsamples+1):
            current = samples[-1]
            proposal, proposal_pdf = self.proposal(current, current_pdf)
            aprob = self.aprob(current, current_pdf, proposal, proposal_pdf)
            
            if aprob > np.random.uniform():
                samples.append(proposal)
                current_pdf = proposal_pdf
            else:
                samples.append(current)
            
            # try to adapt if sampler is adaptive
            if self.is_adaptive:
                self.adapt(t, current, current_pdf, aprob)
            
            if t%1000 == 0:
                print('passed: ', t, 'samples')
        
        return samples[1:]

class AdaptiveMetropolis(StaticMetropolis):
    """
    adaptive Metropolis-Hastings sampler according to Haario (2001)
    """
    
    def __init__(self, ndim, target_pdf, proposal_dist, t_initial, adapt_schedule):
        super().__init__(ndim, target_pdf, proposal_dist, is_adaptive=True)
        
        self.t_initial = t_initial
        self.adapt_schedule = adapt_schedule
        self.s_d = 2.4**2/self.ndim
        self.epsilon = 1e-6
        
        self.mean = None
        self.mean_previous = None
        self.cov = self.proposal_dist.cov
    
    def adapt(self, t, current, current_pdf=None, aprob=None):
        if not type(current) is np.ndarray:
            current = np.array(current)
        if t==1:
            self.mean = current
        else:
            self.mean_previous = self.mean
            self.mean = 1/(t+1) * (t*self.mean + current)
        
        if t > self.t_initial:
            self.cov = (t-1)/t * self.cov + self.s_d/t * (t*np.outer(self.mean_previous, self.mean_previous) - (t+1)*np.outer(self.mean, self.mean) + np.outer(current, current) + self.epsilon*np.identity(self.ndim))
        
            if self.adapt_schedule(t) is True:
                self.proposal_dist.cov = self.cov
            
    
#class RobustAdaptiveMetropolis(StaticMetropolis):
#    """
#    rebustadaptive Metropolis-Hastings sampler according to Vihola (2012)
#    """
    
