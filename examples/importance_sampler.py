from monte_carlo import MonteCarloMultiImportance, MultiChannel, \
    AcceptRejectSampler
from monte_carlo import densities
from monte_carlo.plotting.plot_1d import plot_1d
from monte_carlo.plotting.plot_2d import plot_2d
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

importance = MonteCarloMultiImportance(channels)

t_start = timer()
# integration phase
est, err = importance(target_pdf, [], [noptim], [])

# could sample the channels directly
# sample
# samples = channels.rvs(nsamples)

# generate unweighted events via acceptance/rejection events
# need to keep track of maximum of target (here 1 since mapping is ideal)
sampler = AcceptRejectSampler(target, 1, ndim, channels.rvs, channels)
samples = sampler.sample(nsamples)
# reshuffle  (acceptance rejection is not a Markov chain)
# samples = samples[np.random.choice(np.arange(nsamples),
#                                    nsamples, replace=False)]


t_end = timer()

print("time: ", t_end - t_start)

if ndim == 1:
    plot_1d(samples, target.pdf, mapping_pdf=channels.pdf)
    print("plot 1d done")
elif ndim == 2:
    plot_2d(samples, target.pdf)
