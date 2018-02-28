from samplers.mcmc.importance_sampler import StaticMultiChannelImportanceSampler
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

np.random.seed(1234)

ndim = 1
#ndim = 2
nsamples = 2000

target = Camel()
target_pdf = counted(target.pdf)

#is_proposal_dists = [Gaussian(mu=ndim*[1/5], cov=0.005), Gaussian(mu=ndim*[4/5], cov=0.005)]
is_proposal_dists = [Gaussian(mu=ndim*[1/5], cov=0.005), Gaussian(mu=ndim*[4/5], cov=0.005)]
is_proposal_weights = [0.5, 0.5]
importance_sampler =  StaticMultiChannelImportanceSampler(ndim, target_pdf, is_proposal_dists, is_proposal_weights)

start = 1/3
#start = np.full(ndim, 0.5)
target_pdf.called = 0
t_start = timer()
samples, mean, variance = importance_sampler.sample(nsamples, start)
t_end = timer()

n_target_calls = target_pdf.called

print_statistics(samples, mean, variance, exp_mean=0.5, exp_var=0.0331, exp_var_var=0.000609, runtime=t_end-t_start, n_target_calls=n_target_calls)

if ndim == 1:
    plot_1d(samples, target.pdf, mapping_pdf=importance_sampler.proposal_dist.pdf)
elif ndim == 2:
    plot_2d(samples, target.pdf)
