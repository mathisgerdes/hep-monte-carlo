import numpy as np
from statistics.effective_sample_size import effective_sample_size

def print_statistics(samples, mean, var, exp_mean, exp_var, exp_var_var, runtime, n_target_calls, n_gradient_calls=None, n_accepted = None):
    nsamples = len(samples)
    
    if n_accepted is None:
        n_accepted = 1
        for i in range(1, nsamples):
            if (samples[i] != samples[i-1]).any():
                n_accepted += 1
    
    ESS_mean = effective_sample_size(samples, mean=exp_mean, var=exp_var)
    ESS_var = effective_sample_size((samples-exp_mean)**2, mean=exp_var, var=exp_var_var)
    min_ESS = min(ESS_mean.min(), ESS_var.min())
    
    print('Total wallclock time:', runtime, ' seconds')
    print('Average time per sample:', runtime/nsamples, ' seconds')
    print('Number of accepted points:', n_accepted)
    print('Number of target calls:', n_target_calls)
    if n_gradient_calls is not None:
        print('Number of gradient calls:', n_gradient_calls)
    print('Sampling probability:', n_accepted/n_target_calls)
    print('Mean estimate:', mean)
    print('Variance estimate:', var)
    print('Naive MC standard error:', np.sqrt(var/nsamples))
    print('Effective sample size (mean):', ESS_mean)
    print('Effective sample size (var):', ESS_var)
    print('Minimum effective sample size per second:', min_ESS/runtime)
    print('Effective Standard Error:', np.sqrt(var/min_ESS))
