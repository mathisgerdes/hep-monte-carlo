from hepmc.core.integration import ImportanceMC
from hepmc.core import AcceptRejectSampler
from hepmc.core.densities import ee_qq
from hepmc.core.densities import Rambo
from hepmc.core import phase_space
#from hepmc.plotting.plot_1d import plot_1d
#from hepmc.plotting.plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer


np.random.seed(1234)

nsamples = 2000

ndim = 8
E_CM = 100.
target = ee_qq(E_CM)
target_pdf = target.pdf

proposal = Rambo(2, E_CM)
importance = ImportanceMC(proposal)

t_start = timer()
# integration phase
integration_sample = importance(target, nsamples)
print(integration_sample.integral, integration_sample.integral_err)


mapping = phase_space.Rambo(E_CM, 2)
target = phase_space.MappedDensity(target, mapping, integration_sample.integral)

bound = np.max(integration_sample.function_values / integration_sample.weights)
sampler = AcceptRejectSampler(target, bound / integration_sample.integral, ndim)
sample = sampler.sample(nsamples)
print(sample)
print(mapping.map(sample.data))

# could sample the channels directly
# sample
# samples = channels.rvs(nsamples)

## generate unweighted events via acceptance/rejection events
#bound = np.max(integration_sample.function_values / integration_sample.weights)
#
#sampler = AcceptRejectSampler(target, bound, ndim, proposal.rvs, proposal)
#sample = sampler.sample(nsamples)
##reshuffle  (acceptance rejection is not a Markov chain)
##samples = samples[np.random.choice(np.arange(nsamples),
##                                   nsamples, replace=False)]
#
##sampler = DefaultMetropolis(ndim, target_pdf, channels.proposal, channels.pdf)
##sample = sampler.sample(nsamples, 0.5)
#
#t_end = timer()
#
#print("time: ", t_end - t_start)
#
##if ndim == 1:
##    plot_1d(sample.data, target.pdf, mapping_pdf=channels.pdf)
##    print("plot 1d done")
##elif ndim == 2:
##    plot_2d(sample.data, target.pdf)
