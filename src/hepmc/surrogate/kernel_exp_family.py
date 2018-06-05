from .surrogate_base import StaticSurrogate
from kernel_exp_family.estimators.finite.gaussian import KernelExpFiniteGaussian
from kernel_exp_family.estimators.lite.gaussian import KernelExpLiteGaussian
from kernel_exp_family.estimators.lite.gaussian_low_rank import KernelExpLiteGaussianLowRank

class KernelExpFiniteGaussianSurrogate(StaticSurrogate):
    def __init__(self, ndim, sigma, lmbda, m):
        self.surrogate = KernelExpFiniteGaussian(sigma, lmbda, m, D=ndim)
    
    def train(self, samples):
        self.surrogate.fit(samples)
    
    def log_pdf_gradient(self, x):
        return self.surrogate.grad(x)
    
class KernelExpLiteGaussianSurrogate(StaticSurrogate):
    def __init__(self, ndim, sigma, lmbda, N):
        self.surrogate = KernelExpLiteGaussian(sigma=sigma, lmbda=lmbda, N=N, D=ndim)
    
    def train(self, samples):
        self.surrogate.fit(samples)
    
    def log_pdf_gradient(self, x):
        return self.surrogate.grad(x)
    
class KernelExpLiteGaussianLowRankSurrogate(StaticSurrogate):
    def __init__(self, ndim, sigma, lmbda, N, cg_tol):
        self.surrogate = KernelExpLiteGaussianLowRank(sigma=sigma, lmbda=lmbda, N=N, D=ndim, cg_tol=cg_tol)
    
    def train(self, samples):
        self.surrogate.fit(samples)
    
    def log_pdf_gradient(self, x):
        return self.surrogate.grad(x)
