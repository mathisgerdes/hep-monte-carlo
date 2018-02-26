import numpy as np

class Proposal(object):
    def __init__(self, mu, cov, is_symmetric):
        self.mu = mu
        self.cov = cov
        self.is_symmetric = is_symmetric
    
    def pdf(self, x, mu=None, cov=None):
        raise NotImplementedError()
    
    def log_pdf(self, x):
        return log(self.pdf(x))
    
    def pdf_gradient(self, x):
        raise NotImplementedError()
    
    def log_pdf_gradient(self, x):
        pdf = self.pdf(x)
        if pdf != 0.:
            return self.pdf_gradient(x) / pdf
        else:
            return np.inf
    
    def rvs(self, mu=None, cov=None):
        raise NotImplementedError()

class MultiChannelProposal(Proposal):
    def __init__(self, proposals, weights):
        mu = [proposal.mu for proposal in proposals]
        cov = [proposal.cov for proposal in proposals]
        super().__init__(mu, cov, is_symmetric=False)
        
        self.nchannels = len(proposals)
        self.proposals = proposals
        self.weights = weights
    
    # weighted sum of proposal pdf's
    def pdf(self, x, mu=None, cov=None):
        pdf = 0
        for i, proposal in enumerate(self.proposals):
            pdf += self.weights[i]*proposal.pdf(x)
        
        return pdf
    
    ## pdf of a single channel
    #def pdf_per_channel(self, x, channel, mu=None, cov=None):
    #    return self.proposals[channel].pdf(x, mu, cov)
    
    # randomly choose a channel, taking the weights into account, and sample from its pdf
    def rvs(self, mu=None, cov=None):
        channel = np.random.choice(self.nchannels, p=self.weights)
        return self.proposals[channel].rvs()
