from hepmc.core.integration import ImportanceMC
from hepmc.core import AcceptRejectSampler
from hepmc.core.densities import ee_qq
from hepmc.core.densities import RamboOnDiet
from hepmc.core import phase_space
#from hepmc.plotting.plot_1d import plot_1d
#from hepmc.plotting.plot_2d import plot_2d
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

# t_start = timer()
# integration phase
integration_sample = importance(target, nsamples)
print(integration_sample.integral, integration_sample.integral_err)


mapping = phase_space.RamboOnDiet(E_CM, 2)
target = phase_space.MappedDensity(target, mapping, integration_sample.integral)

bound = np.max(integration_sample.function_values / integration_sample.weights)
sampler = AcceptRejectSampler(target, bound/integration_sample.integral, target.ndim)
sample = sampler.sample(nsamples)
print(sample)
print(mapping.map(sample.data))
