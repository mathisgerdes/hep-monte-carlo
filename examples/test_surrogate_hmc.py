from samplers.mcmc.hmc import StaticHMC
#from samplers.mcmc.hmc import DualAveragingHMC
from surrogates.kernel_exp_family import KernelExpLiteGaussianSurrogate
from densities.camel import UnconstrainedCamel
from proposals.gaussian import IsotropicZeroMeanGaussian
from plotting.plot_1d import plot_1d
#from plotting.plot_2d import plot_2d
import numpy as np

np.seterr(all='warn')
np.random.seed(1234)

ndim = 1
nsamples = 1000
nadapt = 500
ntrain = 500

def hmc_adapt_schedule(t):
    if t > nadapt:
        return False
    else:
        return True

target = UnconstrainedCamel()
train_sampler = StaticHMC(ndim, target.pdf, target.log_pdf_gradient, 0.05, 0.1, 10, 30, momentum=IsotropicZeroMeanGaussian(ndim, 1))
start = np.full(ndim, 0.44)
#train_sampler = DualAveragingHMC(ndim, target.pdf, target.log_pdf_gradient, 1., start, hmc_adapt_schedule, momentum=IsotropicZeroMeanGaussian(ndim, 1))

surrogate = KernelExpLiteGaussianSurrogate(ndim=ndim, sigma=0.5, lmbda=0.0001, N=ntrain)

train_samples = train_sampler.sample(ntrain, start)
print(np.array(train_samples))
print(np.array(train_samples).shape)
#plot_1d([sample[0] for sample in train_samples], target.pdf)
surrogate.train(np.array(train_samples))
print(surrogate.log_pdf_gradient(np.array([1/3])))

sampler = StaticHMC(ndim, target.pdf, surrogate.log_pdf_gradient, 0.05, 0.1, 10, 30, momentum=IsotropicZeroMeanGaussian(ndim, 1))
#sampler = DualAveragingHMC(ndim, target.pdf, surrogate.log_pdf_gradient, 1., start, hmc_adapt_schedule, momentum=IsotropicZeroMeanGaussian(ndim, 1))
#
samples = sampler.sample(nsamples, start)
samples = [sample[0] for sample in samples]
print(samples)
##
plot_1d(samples, target.pdf, target_log_pdf_gradient=target.log_pdf_gradient, surrogate_log_pdf_gradient=surrogate.log_pdf_gradient)
#plot_2d(samples, target.pdf)
