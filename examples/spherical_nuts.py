from samplers.mcmc.spherical_nuts import SphericalNUTS
from densities.camel import UnconstrainedCamel
from plotting.plot_1d import plot_1d
from plotting.plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer
from statistics.print_statistics import print_statistics

# decorator to count calls to target function
def counted(fn):
    def wrapper(*args, **kwargs):
        wrapper.called += 1
        return fn(*args, **kwargs)
    wrapper.called = 0
    wrapper.__name__ = fn.__name__
    return wrapper

nadapt = 1000
def adapt_schedule(t):
    if t > nadapt:
        return False
    else:
        return True

np.random.seed(1234)

ndim = 1
nsamples = 10000
nburnin = nadapt

target = UnconstrainedCamel()
target_log_pdf = counted(target.log_pdf)
target_log_pdf_gradient = counted(target.log_pdf_gradient)

start = np.full(ndim, 1/3)

lim_lower = np.full(ndim, 0)
lim_upper = np.full(ndim, 1)
sampler = SphericalNUTS(ndim, target_log_pdf, target_log_pdf_gradient, start, adapt_schedule, lim_lower, lim_upper, t0=10., gamma=.05)

target_log_pdf.called = 0
target_log_pdf_gradient.called = 0
t_start = timer()
samples, mean, variance = sampler.sample(nsamples, start)
t_end = timer()

# discard burnin samples
samples = samples[1000:]
mean = samples.mean()
variance = samples.var()

n_target_calls = target_log_pdf.called
n_gradient_calls = target_log_pdf_gradient.called

print_statistics(samples, mean, variance, exp_mean=0.5, exp_var=0.0331, exp_var_var=0.000609, runtime=t_end-t_start, n_target_calls=n_target_calls, n_gradient_calls=n_gradient_calls)

if ndim == 1:
    plot_1d(samples, target.pdf)
elif ndim == 2:
    plot_2d(samples, target.pdf)
else:
    plot_1d(samples[:, 0], target.pdf)
