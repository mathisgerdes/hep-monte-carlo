from hepmc.plotting.plot_2d import plot_2d
from hepmc import *
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


target = densities.UnconstrainedCamel(ndim)
#sampler = StaticHMC(ndim, target.pdf, target.log_pdf_gradient, 0.01, 0.1, 10, 150)
sampler = hamiltonian.HamiltonianUpdate(
    target, densities.Gaussian(ndim), 20, 0.05)
start = np.full(ndim, 0.44)
#sampler = DualAveragingHMC(ndim, target.pdf, target.log_pdf_gradient, 1, start, hmc_adapt_schedule, momentum=IsotropicZeroMeanGaussian(ndim, 1))
#sampler = NUTS(ndim, target.pdf, target.log_pdf_gradient, start, hmc_adapt_schedule, momentum=IsotropicZeroMeanGaussian(ndim, 1))

samples = sampler.sample(nsamples, start).data
#samples = [sample[0] for sample in samples]
#print(samples)

plot_2d(samples, target.pdf)
