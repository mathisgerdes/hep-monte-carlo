from samplers.mcmc.importance_sampler import AdaptiveMultiChannelImportanceSampler
from samplers.mcmc.hmc import StaticHMC
from surrogates.kernel_exp_family import KernelExpLiteGaussianSurrogate
from samplers.mcmc.mixed import MixedSampler
from densities.camel import UnconstrainedCamel
from proposals.gaussian import Gaussian
from proposals.gaussian import IsotropicZeroMeanGaussian
from plotting.plot_1d import plot_1d
from plotting.plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer
from statistics.print_statistics import print_statistics

# decorator to count calls to target function
def counted(fn):
    def wrapper(*args, **kwargs):
        wrapper.called += 1
        return fn(*args, **kwargs)
    wrapper.called = 0
    wrapper.__name__ = fn.__name__
    return wrapper

np.random.seed(1234)

#ndim = 1
ndim = 2

nsamples = 2000
ntrain = 1000
nburnin = 1000
nadapt = 1000
t_adapt = [100, 200 ,400, 700]

def hmc_adapt_schedule(t):
    if t > nadapt:
        return False
    else:
        return True

def importance_sampler_adapt_schedule(t):
    if t in t_adapt:
        return True
    else:
        return False

target = UnconstrainedCamel()
target_pdf = counted(target.pdf)

start = np.full(ndim, 0.5)

# initialise importance sampler
is_proposal_dists = [Gaussian(mu=ndim*[1/3], cov=0.005), Gaussian(mu=ndim*[2/3], cov=0.005)]
is_proposal_weights = [0.5, 0.5]
#importance_sampler =  StaticMultiChannelImportanceSampler(ndim, target.pdf, is_proposal_dists, is_proposal_weights)
importance_sampler =  AdaptiveMultiChannelImportanceSampler(ndim, target_pdf, is_proposal_dists, is_proposal_weights, importance_sampler_adapt_schedule)

# burn in of importance sampler
importance_sampler.sample(nburnin, start)
importance_sampler.is_adaptive = False

# generate training samples
train_samples, _, _ = importance_sampler.sample(ntrain, start)

# train the surrogate
surrogate = KernelExpLiteGaussianSurrogate(ndim=ndim, sigma=0.5, lmbda=0.0001, N=ntrain)
surrogate.train(train_samples)

# initialise HMC
hmc_sampler = StaticHMC(ndim, target_pdf, surrogate.log_pdf_gradient, 0.05, 0.1, 10, 30, momentum=IsotropicZeroMeanGaussian(ndim, 1))

# construct mixed sampler
sampler_weights = [0.5, 0.5]
sampler = MixedSampler([hmc_sampler, importance_sampler], sampler_weights)

target_pdf.called = 0
t_start = timer()
samples, mean, variance = sampler.sample(nsamples, start)
t_end = timer()

n_target_calls = target_pdf.called

print_statistics(samples, mean, variance, exp_mean=0.5, exp_var=0.0331, exp_var_var=0.000609, runtime=t_end-t_start, n_target_calls=n_target_calls)

if ndim == 1:
    plot_1d(samples, target.pdf, mapping_pdf=importance_sampler.proposal_dist.pdf, target_log_pdf_gradient=target.log_pdf_gradient, surrogate_log_pdf_gradient=surrogate.log_pdf_gradient)
elif ndim == 2:
    plot_2d(samples, target.pdf)
