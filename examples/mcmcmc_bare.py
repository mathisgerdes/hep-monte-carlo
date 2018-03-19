from samplers.mcmc.metropolis import AdaptiveMetropolis
from samplers.mcmc.importance_sampler import StaticMultiChannelImportanceSampler
from samplers.mcmc.mixed import MixedSampler
from densities.camel import Camel
from proposals.gaussian import Gaussian
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

# set seed to PRNG, so that results are reproducable
np.random.seed(1234)

ndim = 1
#ndim = 2
nsamples = 10000
nburnin = 1000
nadapt = nburnin

# adapt proposal width every iteration until t > nadapt
def metropolis_adapt_schedule(t):
    if t > nadapt:
        return False
    else:
        return True

# target function
target = Camel()
target_pdf = counted(target.pdf)

# initial value
start = np.full(ndim, 0.5)

# proposal function for Metropolis sampler
metropolis_proposal = Gaussian(mu=ndim*[0.5], cov=0.005)

# initialize Metropolis Sampler
metropolis_sampler = AdaptiveMetropolis(ndim, target_pdf, metropolis_proposal, t_initial=100, adapt_schedule=metropolis_adapt_schedule)

# burn-in phase where the Metropolis sampler adapts its proposal width
metropolis_sampler.sample(nburnin, start)

# after that stop adaptation in order to ensure ergodicity of the chain
metropolis_sampler.is_adaptive = False

## importance sampler mapping
# this mapping is perfect and results in 100% efficiency
is_proposal_dists = [Gaussian(mu=ndim*[1/3], cov=0.005), Gaussian(mu=ndim*[2/3], cov=0.005)]
is_proposal_weights = [0.5, 0.5]
# in this mapping the location of the peaks has been shifted so that it is not optimal / more realistic
#is_proposal_dists = [Gaussian(mu=ndim*[1/5], cov=0.005), Gaussian(mu=ndim*[4/5], cov=0.005)]
#is_proposal_weights = [0.5, 0.5]

# initialize the importance sampler
# here we use the non-adaptive version, as we know the optimal channel weights
importance_sampler =  StaticMultiChannelImportanceSampler(ndim, target_pdf, is_proposal_dists, is_proposal_weights)

# initialize the mixed sampler a.k.a. (MC)^3
# the sampler weights (beta parameter) can be tuned
sampler_weights = [0.5, 0.5] # should sum to 1
sampler = MixedSampler([metropolis_sampler, importance_sampler], sampler_weights)

# produce the samples
target_pdf.called = 0
t_start = timer()
samples, mean, variance = sampler.sample(nsamples, start)
t_end = timer()

n_target_calls = target_pdf.called

# print some statistics (can take a while if sample size is large!)
print_statistics(samples, mean, variance, exp_mean=0.5, exp_var=0.0331, exp_var_var=0.000609, runtime=t_end-t_start, n_target_calls=n_target_calls)

# plot the results
if ndim == 1:
    plot_1d(samples, target.pdf, mapping_pdf=importance_sampler.proposal_dist.pdf)
elif ndim == 2:
    plot_2d(samples, target.pdf)
else:
    plot_1d(samples[:, 0], target.pdf)
