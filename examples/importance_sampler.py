from samplers.mcmc.importance_sampler import StaticMultiChannelImportanceSampler
from densities.camel import Camel
from proposals.gaussian import Gaussian
#from plotting.plot_1d import plot_1d
from plotting.plot_2d import plot_2d
import numpy as np

np.random.seed(1234)

ndim = 2
nsamples = 2000

target = Camel()

is_proposal_dists = [Gaussian(mu=ndim*[1/3], cov=0.005), Gaussian(mu=ndim*[2/3], cov=0.005)]
is_proposal_weights = [0.5, 0.5]
importance_sampler =  StaticMultiChannelImportanceSampler(ndim, target.pdf, is_proposal_dists, is_proposal_weights)

start = np.full(ndim, 0.5)
samples = importance_sampler.sample(nsamples, ndim*[0.5])

#plot_1d(samples, target.pdf, mapping_pdf=importance_sampler.proposal_dist.pdf)
plot_2d(samples, target.pdf)
