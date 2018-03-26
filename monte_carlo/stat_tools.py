import numpy as np

def auto_cov(a):
    mean = np.mean(a)
    acov = np.empty(len(a))
    acov[0] = np.var(a)
    for k in range(1, len(a)):
        acov[k] = np.sum((a[:-k]-mean)*(a[k:]-mean)) / len(a)
    return acov

def auto_corr(a):
    acov = auto_cov(a)
    return acov / acov[0]
