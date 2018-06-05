from hepmc import *
from hepmc import densities
from hepmc.plotting.plot_1d import plot_1d
from hepmc.plotting.plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer


np.random.seed(1234)

ndim = 1
#ndim = 2
nopt = 1000
nsamples = 2000

target = densities.Camel(ndim)
target_pdf = target.pdf

is_proposal_weights = np.array([0.5, 0.5])
channels = MultiChannel([
    densities.Gaussian(ndim, mu=1/5, cov=0.005),
    densities.Gaussian(ndim, mu=4/5, cov=0.005)],
    is_proposal_weights)

start = np.full(ndim, 1/3)


importance = MultiChannelMC(channels)

t_start = timer()
# integration phase
integration_sample = importance(target_pdf, [], [nopt], [])

x = np.linspace(0, 1, 100)
bound = np.max(integration_sample.function_values / integration_sample.weights)
sampler = AcceptRejectSampler(target, bound, ndim, channels.rvs, channels)
sample = sampler.sample(nsamples)

t_end = timer()

print("time", t_end - t_start)

if ndim == 1:
    plot_1d(sample.data, target.pdf, mapping_pdf=channels.pdf)
elif ndim == 2:
    plot_2d(sample.data, target.pdf)
