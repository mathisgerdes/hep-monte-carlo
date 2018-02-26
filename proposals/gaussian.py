from proposals.proposal_base import Proposal
#from scipy.stats import norm, multivariate_normal
from scipy.stats import multivariate_normal
import numpy as np

class Gaussian(Proposal):
    def __init__(self, mu, cov):
        super().__init__(mu, cov, is_symmetric=True)
    
    def pdf(self, x, mu=None, cov=None):
        if mu is None:
            mu = self.mu
        if cov is None:
            cov = self.cov
            
        return multivariate_normal.pdf(x, mu, cov)
    
    def rvs(self, mu=None, cov=None):
        if mu is None:
            mu = self.mu
        if cov is None:
            cov = self.cov
            
        return multivariate_normal.rvs(mu, cov)

class IsotropicZeroMeanGaussian(Proposal):
    def __init__(self, ndim, cov=1):
        mu = 0
        super().__init__(mu, cov, is_symmetric=True)
        self.ndim = ndim
        
    def pdf(self, x):
        return multivariate_normal.pdf(x, mean=np.zeros(self.ndim), cov=np.sqrt(self.cov))
    
    def log_pdf(self, x):
        D = len(x)
        const_part = -0.5 * D * np.log(2 * np.pi)
        quadratic_part = -np.dot(x, x) / (2 * (self.cov))
        log_determinant_part = -0.5 * D * np.log(self.cov)
        return const_part + log_determinant_part + quadratic_part
    
    def log_pdf_gradient(self, x):
        return -x / self.cov
    
    def rvs(self):
        return np.random.randn(self.ndim) * np.sqrt(self.cov)
