#sampler
from samplers.mcmc.metropolis import AdaptiveMetropolis
from samplers.mcmc.importance_sampler import AdaptiveMultiChannelImportanceSampler
from samplers.mcmc.hmc import StaticHMC

from samplers.mcmc.mixed import MixedSampler


from surrogates.kernel_exp_family import KernelExpLiteGaussianSurrogate

#targets and proposals
from densities.camel import Camel
from proposals.gaussian import Gaussian
from proposals.gaussian import IsotropicZeroMeanGaussian



#statistics
from timeit import default_timer as timer
from statistics.print_statistics import print_statistics
from testsuit import testsuit

#setup
from SetUp import MixedSamplerSetUp

#basics
import numpy as np

# decorator to count calls to target function
def counted(fn):
    def wrapper(*args, **kwargs):
        wrapper.called += 1
        return fn(*args, **kwargs)
    wrapper.called = 0
    wrapper.__name__ = fn.__name__
    return wrapper



# general setup
np.random.seed(1234)

ndim = 2
nsamples = 100000
nburnin = 1000
ntrain = 1000
nadapt = 1000
t_adapt = [100, 200 ,400, 700]
beta = [0.1,0.2,0.3,0.4,0.5,0.6,0.7,.8,0.9]

target = Camel()
target_pdf = counted(target.pdf)
targetName = "Camel"


if ndim == 1:
    start = 0.5
else:
    start = np.full(ndim,0.5)



# config importance sampler
def importance_sampler_adapt_schedule(t):
    return  t in t_adapt

is_proposal_dists = [Gaussian(mu=ndim*[1/3], cov=0.005), Gaussian(mu=ndim*[2/3], cov=0.005)]
is_proposal_weights = [0.5, 0.5]

is_args = {
            "ndim":ndim,
            "target_pdf":target_pdf,
            "proposal_dists":is_proposal_dists,
            "initial_weights":is_proposal_weights,
            "adapt_schedule":importance_sampler_adapt_schedule
            }

# config metropolis
def metropolis_adapt_schedule(t):
    return t <= nadapt

metropolis_proposal = Gaussian(mu=ndim*[0.5], cov=0.005)
mp_args = {
            "ndim":ndim,
            "target_pdf":target_pdf,
            "proposal_dist":metropolis_proposal,
            "t_initial":100,
            "adapt_schedule":metropolis_adapt_schedule
            }

# config hm w s
def hmc_adapt_schedule(t):
    return t <= nadapt

importance_sampler =  AdaptiveMultiChannelImportanceSampler(ndim, target_pdf, is_proposal_dists, is_proposal_weights, importance_sampler_adapt_schedule)

# burn in of importance sampler
importance_sampler.sample(nburnin, start)
importance_sampler.is_adaptive = False

# generate training samples
train_samples, _, _ = importance_sampler.sample(ntrain, start)

# train the surrogate
surrogate = KernelExpLiteGaussianSurrogate(ndim=ndim, sigma=0.5, lmbda=0.0001, N=ntrain)
surrogate.train(train_samples)

hm_args = {
            "ndim":ndim,
            "target_pdf":target_pdf,
            "target_log_pdf_gradient":surrogate.log_pdf_gradient,
            "stepsize_min":0.05,
            "stepsize_max":0.1,
            "nsteps_min":10,
            "nsteps_max":30,
            "momentum":IsotropicZeroMeanGaussian(ndim,1)
          }





##################################

# looping parameters
betas=[0.1,0.2,0.3,0.4,0.5,0.6,0.7,0.8,0.9]

for beta in betas:
    print(beta)
    np.random.seed(1234)
    #sampler = MixedSamplerSetUp([AdaptiveMetropolis,AdaptiveMultiChannelImportanceSampler],[mp_args,is_args],beta=beta,nburnin=1000,start=start)
    sampler = MixedSamplerSetUp([StaticHMC,AdaptiveMultiChannelImportanceSampler],[hm_args,is_args],beta=beta,nburnin=1000,start=start)


    target_pdf.called = 0

    
    t_start = timer()
    samples, mean, variance = sampler.sample(nsamples, start)
    t_end = timer()
    n_target_calls = target_pdf.called


    SAMPLE = {
                "samples":samples,
                "mean":mean,
                "variance":variance,
                "runtime":t_end-t_start,
                "n_target_calls":n_target_calls
            }
    SampleName=targetName+"_"+str(nsamples/1000)+"k"+"_hmc-beta_"+str(beta) 
    np.save(SampleName,SAMPLE)
   
