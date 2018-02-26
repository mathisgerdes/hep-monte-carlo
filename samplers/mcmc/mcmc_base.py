from samplers.sampler_base import Sampler
import numpy as np

class MCMC(Sampler):
    def __init__(self, ndim, target_pdf, proposal_dist, is_adaptive):
        super().__init__(ndim, target_pdf, is_adaptive)
        self.proposal_dist = proposal_dist
    
    def proposal(self, current, current_pdf):
        raise NotImplementedError()
    
    def aprob(self, current, current_pdf, proposal, proposal_pdf):
        if self.proposal_dist.is_symmetric is True:
            return min(1, proposal_pdf/current_pdf)
        
        else:
            return min(1, proposal_pdf*self.proposal_dist.pdf(current) / (current_pdf*self.proposal_dist.pdf(proposal)))

    def step(self, current, current_pdf):
        proposal, proposal_pdf = self.proposal(current, current_pdf)
        aprob = self.aprob(current, current_pdf, proposal, proposal_pdf)
        
        if aprob > np.random.uniform():
            return proposal, proposal_pdf
        
        else:
            return current, current_pdf
