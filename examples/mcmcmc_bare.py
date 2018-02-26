from samplers.mcmc.metropolis import AdaptiveMetropolis
from samplers.mcmc.importance_sampler import AdaptiveMultiChannelImportanceSampler
from samplers.mcmc.mixed import MixedSampler
from densities.camel import Camel
from proposals.gaussian import Gaussian
from plotting.plot_1d import plot_1d
from plotting.plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer

# decorator to count calls to target function
def counted(fn):
    def wrapper(*args, **kwargs):
        wrapper.called += 1
        return fn(*args, **kwargs)
    wrapper.called = 0
    wrapper.__name__ = fn.__name__
    return wrapper

np.random.seed(1234)

ndim = 1
#ndim = 2
nsamples = 2000
nburnin = 1000
nadapt = 1000
t_adapt = [100, 200 ,400, 700]

def metropolis_adapt_schedule(t):
    if t > nadapt:
        return False
    else:
        return True

def importance_sampler_adapt_schedule(t):
    if t in t_adapt:
        return True
    else:
        return False

target = Camel()
target_pdf = counted(target.pdf)

if ndim == 1:
    start = 0.5
else:
    start = np.full(ndim, 0.5)

metropolis_proposal = Gaussian(mu=ndim*[0.5], cov=0.005)
#metropolis_sampler = StaticMetropolis(ndim, target_pdf, metropolis_proposal)
metropolis_sampler = AdaptiveMetropolis(ndim, target_pdf, metropolis_proposal, t_initial=100, adapt_schedule=metropolis_adapt_schedule)

# burn in
metropolis_sampler.sample(nburnin, start)
metropolis_sampler.is_adaptive = False

is_proposal_dists = [Gaussian(mu=ndim*[1/3], cov=0.005), Gaussian(mu=ndim*[2/3], cov=0.005)]
is_proposal_weights = [0.5, 0.5]
#importance_sampler =  StaticMultiChannelImportanceSampler(ndim, target_pdf, is_proposal_dists, is_proposal_weights)
importance_sampler =  AdaptiveMultiChannelImportanceSampler(ndim, target_pdf, is_proposal_dists, is_proposal_weights, importance_sampler_adapt_schedule)

# burn in
importance_sampler.sample(nburnin, start)
importance_sampler.is_adaptive = False

sampler_weights = [0.5, 0.5]
sampler = MixedSampler([metropolis_sampler, importance_sampler], sampler_weights)

t_start = timer()
samples = sampler.sample(nsamples, start)
t_end = timer()

n_target_calls = target_pdf.called

n_accepted = 1
for i in range(1, nsamples):
    if samples[i] != samples[i-1]:
        n_accepted += 1

print('Total wallclock time: ', t_end-t_start, ' seconds')
print('Average time per sample: ', (t_end-t_start)/nsamples, ' seconds')
print('Number of accepted points: ', n_accepted)
print('Number of target calls: ', n_target_calls)
print('Sampling probability: ', n_accepted/n_target_calls)


if ndim == 1:
    plot_1d(samples, target.pdf, mapping_pdf=importance_sampler.proposal_dist.pdf)
elif ndim == 2:
    plot_2d(samples, target.pdf)
