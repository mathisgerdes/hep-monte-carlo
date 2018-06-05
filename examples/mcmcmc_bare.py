from hepmc import *
from hepmc.plotting.plot_1d import plot_1d
from hepmc.plotting.plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer

# set seed to PRNG, so that results are reproducible
np.random.seed(1234)

ndim = 1  # 2
nsamples = 10000
nburnin = 1000
nadapt = nburnin


# adapt proposal width every iteration until t > nadapt
def metropolis_adapt_schedule(t):
    return t <= nadapt


# target function
target = densities.Camel(ndim)
util.count_calls(target, 'pdf')

# initial value
start = np.full(ndim, 0.5)

# proposal function for Metropolis sampler
metropolis_proposal = densities.Gaussian(ndim, mu=0.5, cov=0.005)

# initialize Metropolis Sampler
metropolis_sampler = AdaptiveMetropolisUpdate(
    ndim, target.pdf, metropolis_proposal,
    t_initial=100, adapt_schedule=metropolis_adapt_schedule)

# burn-in phase where the Metropolis sampler adapts its proposal width
metropolis_sampler.sample(nburnin, start)

# after that stop adaptation in order to ensure ergodicity of the chain
metropolis_sampler.is_adaptive = False

# importance sampler mapping
# this mapping is perfect and results in 100% efficiency
channels = MultiChannel([
    densities.Gaussian(ndim, mu=1/3, cov=0.005),
    densities.Gaussian(ndim, mu=2/3, cov=0.005)])

# in this mapping the location of the peaks has been shifted so that it is not optimal / more realistic
# is_proposal_dists = [Gaussian(mu=ndim*[1/5], cov=0.005), Gaussian(mu=ndim*[4/5], cov=0.005)]

# initialize the importance sampler
importance_sampler = DefaultMetropolis(ndim, target.pdf, channels)

# initialize the mixed sampler a.k.a. (MC)^3
# the sampler weights (beta parameter) can be tuned
sampler_weights = [0.5, 0.5]  # should sum to 1
sampler = MixingMarkovUpdate(ndim, [metropolis_sampler, importance_sampler],
                             sampler_weights)

# produce the samples
target.pdf.count = 0

t_start = timer()
sample = sampler.sample(nsamples, start)
t_end = timer()

n_target_calls = target.pdf.count

# print some statistics (can take a while if sample size is large!)
print('time: ', t_end - t_start)
print('pdf calls: ', n_target_calls)
print(sample)

# plot the results
if ndim == 1:
    plot_1d(sample.data, target.pdf, mapping_pdf=channels.pdf)
elif ndim == 2:
    plot_2d(sample.data, target.pdf)
else:
    plot_1d(sample.data[:, 0], target.pdf)
