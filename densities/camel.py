from densities.density_base import Density
from scipy.stats import multivariate_normal
import numpy as np

class UnconstrainedCamel(Density):
    """
    sum of two gaussians on a [0, 1] hypercube
    """
    def __init__(self, mu_a=1/3, mu_b=2/3, a=0.1):
        self.mu_a = mu_a
        self.mu_b = mu_b
        self.cov = a**2/2
    
    def pdf(self, x):
        if not type(x) is np.ndarray:
            x = np.array(x)
        
        ndim = x.size
        mu_a = ndim*[self.mu_a]
        mu_b = ndim*[self.mu_b]
        return (multivariate_normal.pdf(x, mean=mu_a, cov=self.cov) + multivariate_normal.pdf(x, mean=mu_b, cov=self.cov)) / 2
    
    def pdf_gradient(self, x):
        if not type(x) is np.ndarray:
            x = np.array(x)
        
        ndim = x.size
        mu_a = ndim*[self.mu_a]
        mu_b = ndim*[self.mu_b]
        return ((-x+self.mu_a)/self.cov * multivariate_normal.pdf(x, self.mu_a, self.cov) + (-x+self.mu_b)/self.cov * multivariate_normal.pdf(x, self.mu_b, self.cov)) / 2
    
class Camel(UnconstrainedCamel):
    """
    sum of two gaussians on a [0, 1] hypercube
    """
    def pdf(self, x):
        if not type(x) is np.ndarray:
            x = np.array(x)
        if not ((x > 0).all() and (x < 1).all()):
            return 0
        
        return super().pdf(x)
    
    def log_pdf(self, x):
        if not type(x) is np.ndarray:
            x = np.array(x)
        if not ((x > 0).all() and (x < 1).all()):
            raise Exception("Out of boundaries: %s" % x)
        
        return super().log_pdf(x)
    
    def pdf_gradient(self, x):
        if not type(x) is np.ndarray:
            x = np.array(x)
        if not ((x > 0).all() and (x < 1).all()):
            raise Exception("Out of boundaries: %s" % x)
        
        return super().pdf_gradient(x)
    
    def log_pdf_gradient(self, x):
        if not type(x) is np.ndarray:
            x = np.array(x)
        if not ((x > 0).all() and (x < 1).all()):
            raise Exception("Out of boundaries: %s" % x)
        
        return super().log_pdf_gradient(x)
