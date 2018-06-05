from hepmc import hamiltonian, densities
from hepmc.plotting.plot_1d import plot_1d
from hepmc.plotting.plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer


nadapt = 100


def adapt_schedule(t):
    return t <= nadapt


np.random.seed(1234)

ndim = 1
nsamples = 10000
nburnin = nadapt

target = densities.UnconstrainedCamel(1)

start = np.full(ndim, 1/3)

lim_lower = np.full(ndim, 0)
lim_upper = np.full(ndim, 1)
#sampler = StaticSphericalHMC(ndim, target_log_pdf, target.log_pdf_gradient, .005, .005, 8, 8, lim_lower, lim_upper)
sampler = hamiltonian.DualAveragingSphericalHMC(
    target, .04, adapt_schedule, lim_lower, lim_upper, t0=10., gamma=.05,
    kappa=.75)

t_start = timer()
sample = sampler.sample(nsamples, start, log_every=1000)
t_end = timer()

# discard burnin samples
sample.data = sample.data[1000:]
#print_statistics(samples, mean, variance, exp_mean=0.5, exp_var=0.0331, exp_var_var=0.000609, runtime=t_end-t_start, n_target_calls=n_target_calls, n_gradient_calls=n_gradient_calls, n_accepted=n_accepted)

if ndim == 1:
    plot_1d(sample.data, target.pdf)
elif ndim == 2:
    plot_2d(sample.data, target.pdf)
else:
    plot_1d(sample.data[:, 0], target.pdf)
