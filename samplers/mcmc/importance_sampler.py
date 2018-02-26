from samplers.mcmc.mcmc_base import MCMC
from proposals.proposal_base import MultiChannelProposal
import numpy as np

class StaticImportanceSampler(MCMC):
    """
    single channel importance sampler
    """
    
    def __init__(self, ndim, target_pdf, proposal_dist, is_adaptive=False):
        super().__init__(ndim, target_pdf, proposal_dist, is_adaptive)
        
    def proposal(self, current, current_pdf):
        proposal = self.proposal_dist.rvs()
        proposal_pdf = self.target_pdf(proposal)
        
        return proposal, proposal_pdf

class StaticMultiChannelImportanceSampler(StaticImportanceSampler):
    """
    multi-channel importance sampler
    """
    
    def __init__(self, ndim, target_pdf, proposal_dists, proposal_weights, is_adaptive=False):
        proposal_dist = MultiChannelProposal(proposal_dists, proposal_weights)
        super().__init__(ndim, target_pdf, proposal_dist, is_adaptive)

class AdaptiveMultiChannelImportanceSampler(StaticMultiChannelImportanceSampler):
    """
    adaptive multi-channel importance sampler
    (adapts the channel weights)
    """
    
    def __init__(self, ndim, target_pdf, proposal_dists, initial_weights, adapt_schedule):
        super().__init__(ndim, target_pdf, proposal_dists, initial_weights, is_adaptive=True)
        self.adapt_schedule = adapt_schedule
        
        self.variance_grad = len(proposal_dists)*[None]
        
    def adapt(self, t, current, current_pdf, previous, previous_pdf):
        weight = current_pdf / self.proposal_dist.pdf(current)
        if t==1:
            self.variance_grad = [proposal.pdf(current)/self.proposal_dist.pdf(current) * weight**2 for proposal in self.proposal_dist.proposals]
        
        else:
            self.variance_grad = [1/(t+1) * (t*variance_grad_i + proposal.pdf(current)/self.proposal_dist.pdf(current) * weight**2) for variance_grad_i, proposal in zip(self.variance_grad, self.proposal_dist.proposals)]
        
        if self.adapt_schedule(t) is True:
            new_weights = [old_weight*np.sqrt(variance_grad_i) for old_weight, variance_grad_i in zip(self.proposal_dist.weights, self.variance_grad)]
            norm = sum(new_weights)
            new_weights = [weight/norm for weight in new_weights]
            self.proposal_dist.weights = new_weights
            
            #reset
            self.variance_grad = [proposal.pdf(current)/self.proposal_dist.pdf(current) * weight**2 for proposal in self.proposal_dist.proposals]
