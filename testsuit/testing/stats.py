import numpy as np
from scipy.stats import chi2


def chiq(a,b):
    b *= np.sum(a)/np.sum(b)
    valid =b>2
    ndof = np.sum(valid)-1
    a = a[valid]
    b = b[valid]
    b *= np.sum(a)/np.sum(b)
    chiq = np.sum((a-b)**2/b)
    p    = 1-chi2.cdf(chiq,ndof)
    return chiq,p

def chiqp(a,b):
    ndim = len(np.shape(a))
    axes = np.arange(ndim)
    chiqp = []
    for axis in axes:
        pa = np.sum(a,axis=tuple(axes[axes!=axis]))
        pb = np.sum(b,axis=tuple(axes[axes!=axis]))
        pb *= np.sum(pa)/np.sum(pb)
        valid = pb >2
        ndof = np.sum(valid)-1
        pa = pa[valid]
        pb = pb[valid]
        pb *= np.sum(pa)/np.sum(pb)
        chiq = np.sum(((pa-pb)**2)/pb)
        p    = 1-chi2.cdf(chiq,ndof)
        chiqp.append(np.array([chiq,p]))
    return np.array(chiqp)



def ks(a,b,alpha=0.05):
    ndim = 1
    if isinstance(a[0],np.ndarray):
        ndim = len(np.shape(a))
    axes = np.arange(ndim)
    ks= []
    for axis in range(ndim):
        n_a = np.sum(a)
        n_b = np.sum(b)
        ps = np.sum(a,axis=tuple(axes[axes!=axis]))
        pt = np.sum(b,axis=tuple(axes[axes!=axis]))

        cps = np.cumsum(ps)/n_a
        cpt = np.cumsum(pt)/n_b
        d_max = np.max(abs(cps-cpt))
        d_alpha = np.sqrt(np.log(2/alpha)/(2*n_a))
        ks.append([d_max,d_alpha])
    return ks

