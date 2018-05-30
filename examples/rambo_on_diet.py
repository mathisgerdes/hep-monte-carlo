from monte_carlo.core.integration import ImportanceMC
from monte_carlo.core import AcceptRejectSampler
from monte_carlo.core.densities import ee_qq
from monte_carlo.core.densities import RamboOnDiet
from monte_carlo.core import phase_space
#from monte_carlo.plotting.plot_1d import plot_1d
#from monte_carlo.plotting.plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer


np.random.seed(1234)

nsamples = 2000

ndim = 2
E_CM = 100.
target = ee_qq(E_CM)
target_pdf = target.pdf

proposal = RamboOnDiet(2, E_CM)
importance = ImportanceMC(proposal)

t_start = timer()
# integration phase
integration_sample = importance(target, nsamples)
print(integration_sample.integral, integration_sample.integral_err)


#target.mapping = phase_space.map_rambo_on_diet
#
#bound = np.max(integration_sample.function_values / integration_sample.weights)
#sampler = AcceptRejectSampler(target, bound, ndim)
#sample = sampler.sample(nsamples)
#print(sample)
#print(target.mapping(sample.data, target.E_CM))
