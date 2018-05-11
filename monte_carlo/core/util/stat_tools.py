"""
Module provides methods for analyzing the statistics of a sample
(or values) generated with Monte Carlo techniques.
"""

import numpy as np
from scipy import stats
# from ..integration import ImportanceMC
# from ..densities import Uniform


def lag_auto_cov(values, k, mean=None):
    if mean is None:
        mean = np.mean(values)
    return np.sum((values[:-k] - mean) * (values[k:] - mean)) / values.shape[0]


def auto_cov(values):
    """ Compute the lag-autocovariance of a given array of values.

    :param values: Array of values.
    :return: Numpy array containing at index k the k-lag autocovariance.
    """
    mean = np.mean(values)
    size = len(values)
    acov = np.empty(size)
    acov[0] = np.var(values)
    for k in range(1, len(values)):
        acov[k] = lag_auto_cov(values, k, mean=mean)
    return acov


def auto_corr(values):
    """ Compute the autocorrelation of a given array of values. """
    acov = auto_cov(values)
    return acov / acov[0]


def fd_bins(sample):
    mins = np.min(sample.data, axis=0)
    maxs = np.max(sample.data, axis=0)
    # h=2IQR(x)N^{−1/3} -> N^{-1/(2 + d)}
    widths = (stats.iqr(sample.data.transpose(), axis=1) *
                  sample.size ** (- 1 / (2 + sample.ndim)))
    bins = np.ceil((maxs - mins) / widths).astype(np.int)
    bins = np.maximum(1, bins)
    return np.asanyarray(bins).astype(np.int)


def bin_wise_chi2(sample, bins=None, min_count=10, nintegral=100):
    """ Compute the bin-wise chi^2 / dof value for a given number of bins.

    If the distribution of points in each bin follows a Poisson distribution
    with the expectation value according to the probability distribution,
    the returned value follows a chi squared distribution with expectation
    value 1.

    Use Freedman Diaconis Estimator with scaling like in sec 3.4 Eqn 3.61 (p83):
        Scott, D.W. (1992),
        Multivariate Density Estimation: Theory, Practice, and Visualization,
    """
    if bins is None:
        bins = fd_bins(sample)
    count, edges = np.histogramdd(sample.data, bins)  # type: np.ndarray, tuple
    relevant = np.where(count >= min_count)

    expected = np.empty(len(relevant[0]))

    vol = np.prod([edges[d][1] - edges[d][0] for d in range(sample.ndim)])
    i = 0
    for mi in zip(*relevant):
        low = [edges[d][mi[d]] for d in range(sample.ndim)]
        high = [edges[d][mi[d]+1] for d in range(sample.ndim)]
        expected[i] = np.mean(sample.target.pdf(np.random.uniform(
            low, high, (nintegral, sample.ndim)))) * sample.size * vol
        i += 1
    finals = np.where(expected >= min_count)[0]
    f_obs = count[relevant][finals]
    chi2, p = stats.chisquare(f_obs, f_exp=expected[finals])
    if len(f_obs > 0):
        return chi2 / (f_obs.size - 1), p
    return None, None
