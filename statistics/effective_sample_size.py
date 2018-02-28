import numpy as np

def effective_sample_size(samples, mean, var):
    shape = samples.shape
    nsamples = shape[0]
    ndim = shape[1]
    sum = np.zeros(ndim)
    for dim in range(ndim):
        lag = 1
        rho = autocorr(samples[:, dim], mean, var, lag)
        while rho >= 0.05:
            sum[dim] = sum[dim] + (1-lag/nsamples)*rho
            lag = lag + 1
            rho = autocorr(samples[:, dim], mean, var, lag)
        
    return nsamples / (1 + 2*sum)

def autocorr(samples, mean, var, lag=1):
    nsamples = len(samples)
    sum = 0
    for m in range(lag, nsamples):
        sum = sum + (samples[m] - mean) * (samples[m-lag] - mean)
        
    return sum / (var*(nsamples-lag))
