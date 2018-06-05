from hepmc import *
from hepmc import densities
from hepmc.plotting.plot_1d import plot_1d
from hepmc.plotting.plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer


np.random.seed(1234)

ndim = 1
#ndim = 2
noptim = 1000
nsamples = 2000

target = densities.Camel(ndim)
target_pdf = target.pdf

is_proposal_weights = np.array([0.5, 0.5])
channels = MultiChannel([
    densities.Gaussian(ndim, mu=1/3, cov=0.005),
    densities.Gaussian(ndim, mu=2/3, cov=0.005)],
    is_proposal_weights)

importance = MultiChannelMC(channels)

t_start = timer()
# integration phase
integration_sample = importance(target_pdf, [], [noptim], [])

# could sample the channels directly
# sample
# samples = channels.rvs(nsamples)

# generate unweighted events via acceptance/rejection events
bound = np.max(integration_sample.function_values / integration_sample.weights)

# sampler = AcceptRejectSampler(target, 1, ndim, channels.rvs, channels)
# sample = sampler.sample(nsamples)
# reshuffle  (acceptance rejection is not a Markov chain)
# samples = samples[np.random.choice(np.arange(nsamples),
#                                    nsamples, replace=False)]

sampler = DefaultMetropolis(ndim, target_pdf, channels.proposal, channels.pdf)
sample = sampler.sample(nsamples, 0.5)

t_end = timer()

print("time: ", t_end - t_start)

if ndim == 1:
    plot_1d(sample.data, target.pdf, mapping_pdf=channels.pdf)
    print("plot 1d done")
elif ndim == 2:
    plot_2d(sample.data, target.pdf)
