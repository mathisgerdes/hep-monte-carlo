import numpy as np
from .helper import interpret_array

def effective_sample_size(sample, mean, var):
    mean = interpret_array(mean, sample.ndim)
    var = interpret_array(var, sample.ndim)
    sum = np.zeros(sample.ndim)
    for dim in range(sample.ndim):
        lag = 1
        rho = autocorr(sample.data[:, dim], sample.size, mean[:, dim], var[:, dim], lag)
        while rho >= 0.05 and lag < sample.size-1:
            sum[dim] = sum[dim] + (1-lag/sample.size)*rho
            lag = lag + 1
            rho = autocorr(sample.data[:, dim], sample.size, mean[:, dim], var[:, dim], lag)
        
    return sample.size / (1 + 2*sum)


def autocorr(sample, size, mean, var, lag=1):
    sum = 0
    for m in range(lag, size):
        sum = sum + (sample[m] - mean) * (sample[m-lag] - mean)
        
    return sum / (var*(size-lag))
