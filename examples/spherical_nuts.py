from monte_carlo import densities, SphericalNUTS
from monte_carlo.plotting.plot_1d import plot_1d
from monte_carlo.plotting.plot_2d import plot_2d
import numpy as np
from timeit import default_timer as timer


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


target = densities.UnconstrainedCamel(ndim)
target_log_pdf = target.pot
target_log_pdf_gradient = target.pot_gradient

start = np.full(ndim, 1/3)

lim_lower = np.full(ndim, 0)
lim_upper = np.full(ndim, 1)
sampler = SphericalNUTS(target, adapt_schedule, lim_lower, lim_upper,
                        t0=10., gamma=.05)
sampler.init_sampler(start, log_every=500)

t_start = timer()
samples = sampler.sample(nsamples)
t_end = timer()

# discard burn-in samples
samples = samples[1000:]
mean = samples.mean()
variance = samples.var()


if ndim == 1:
    plot_1d(samples, target.pdf)
elif ndim == 2:
    plot_2d(samples, target.pdf)
else:
    plot_1d(samples[:, 0], target.pdf)

print("time", t_end - t_start)
