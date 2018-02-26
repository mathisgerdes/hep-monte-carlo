#from samplers.mcmc.metropolis import StaticMetropolis
from samplers.mcmc.metropolis import AdaptiveMetropolis
#from samplers.mcmc.importance_sampler import StaticMultiChannelImportanceSampler
from samplers.mcmc.importance_sampler import AdaptiveMultiChannelImportanceSampler
from samplers.mcmc.mixed import MixedSampler
from densities.camel import Camel
from proposals.gaussian import Gaussian
#from plotting.plot_1d import plot_1d
from plotting.plot_2d import plot_2d
from numpy.random import seed

seed(1234)

ndim = 2
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
metropolis_proposal = Gaussian(mu=ndim*[0.5], cov=0.005)
#metropolis_sampler = StaticMetropolis(ndim, target.pdf, metropolis_proposal)
metropolis_sampler = AdaptiveMetropolis(ndim, target.pdf, metropolis_proposal, t_initial=100, adapt_schedule=metropolis_adapt_schedule)

# burn in
metropolis_sampler.sample(nburnin, ndim*[0.5])
metropolis_sampler.is_adaptive = False

is_proposal_dists = [Gaussian(mu=ndim*[1/3], cov=0.005), Gaussian(mu=ndim*[2/3], cov=0.005)]
is_proposal_weights = [0.5, 0.5]
#importance_sampler =  StaticMultiChannelImportanceSampler(ndim, target.pdf, is_proposal_dists, is_proposal_weights)
importance_sampler =  AdaptiveMultiChannelImportanceSampler(ndim, target.pdf, is_proposal_dists, is_proposal_weights, importance_sampler_adapt_schedule)

# burn in
importance_sampler.sample(nburnin, ndim*[0.5])
importance_sampler.is_adaptive = False

sampler_weights = [0.5, 0.5]
sampler = MixedSampler([metropolis_sampler, importance_sampler], sampler_weights)

samples = sampler.sample(nsamples, ndim*[0.5])
print(metropolis_proposal.mu)
print(metropolis_proposal.cov)
print(importance_sampler.proposal_dist.weights)

#plot_1d(samples, target.pdf, mapping_pdf=importance_sampler.proposal_dist.pdf)
plot_2d(samples, target.pdf)
