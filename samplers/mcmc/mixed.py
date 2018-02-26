from samplers.mcmc.mcmc_base import MCMC
import numpy as np

class MixedSampler(MCMC):
    def __init__(self, samplers, weights):
        ndim = samplers[0].ndim
        target_pdf = samplers[0].target_pdf
        proposal_dist = [sampler.proposal_dist for sampler in samplers]
        is_adaptive = any([sampler.is_adaptive for sampler in samplers])
        super().__init__(ndim, target_pdf, proposal_dist, is_adaptive)
        
        self.samplers = samplers
        self.weights = weights
        
    def step(self, current, current_pdf):
        sampler = np.random.choice(self.samplers, p=self.weights)
        return sampler.step(current, current_pdf)
    
    def adapt(self, t, sample, sample_pdf):
        for sampler in self.samplers:
            if sampler.is_adaptive:
                sampler.adapt(t, sample, sample_pdf)
