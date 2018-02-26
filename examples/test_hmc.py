from samplers.mcmc.hmc import StaticHMC
#from samplers.mcmc.hmc import DualAveragingHMC
#from samplers.mcmc.nuts import NUTS
from densities.camel import UnconstrainedCamel
from proposals.gaussian import IsotropicZeroMeanGaussian
#from plotting.plot_1d import plot_1d
from plotting.plot_2d import plot_2d
import numpy as np

np.seterr(all='warn')
np.random.seed(1234)

ndim = 2
nsamples = 1000
nadapt = 500

def hmc_adapt_schedule(t):
    if t > nadapt:
        return False
    else:
        return True

target = UnconstrainedCamel()
#sampler = StaticHMC(ndim, target.pdf, target.log_pdf_gradient, 0.01, 0.1, 10, 150)
sampler = StaticHMC(ndim, target.pdf, target.log_pdf_gradient, 0.05, 0.1, 10, 30, momentum=IsotropicZeroMeanGaussian(ndim, 1))
start = np.full(ndim, 0.44)
#sampler = DualAveragingHMC(ndim, target.pdf, target.log_pdf_gradient, 1, start, hmc_adapt_schedule, momentum=IsotropicZeroMeanGaussian(ndim, 1))
#sampler = NUTS(ndim, target.pdf, target.log_pdf_gradient, start, hmc_adapt_schedule, momentum=IsotropicZeroMeanGaussian(ndim, 1))

samples = sampler.sample(nsamples, start)
#samples = [sample[0] for sample in samples]
#print(samples)

plot_2d(samples, target.pdf)
