from hepmc import densities, AdaptiveMetropolisUpdate
from hepmc.plotting.plot_1d import plot_1d
from hepmc.plotting.plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer

np.random.seed(1234)

# ndim = 1
ndim = 1
nsamples = 2000
nburnin = 1000
nadapt = 1000


def metropolis_adapt_schedule(t):
    return t <= nadapt


target = densities.Camel(ndim)
target_pdf = target.pdf

start = np.full(ndim, 0.5)

metropolis_proposal = densities.Gaussian(ndim, mu=ndim*[0.5], cov=0.005)
metropolis_sampler = AdaptiveMetropolisUpdate(
    ndim, target_pdf, metropolis_proposal, t_initial=100,
    adapt_schedule=metropolis_adapt_schedule)

# burn in
metropolis_sampler.sample(nburnin, start)
metropolis_sampler.is_adaptive = False

t_start = timer()
sample = metropolis_sampler.sample(nsamples, start)
t_end = timer()
print(t_end - t_start)


if ndim == 1:
    plot_1d(sample.data, target.pdf)
elif ndim == 2:
    plot_2d(sample.data, target.pdf)
