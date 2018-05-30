"""
Module provides methods for analyzing the statistics of a sample
(or values) generated with Monte Carlo techniques.
"""

import numpy as np
from scipy import stats
from .helper import interpret_array


def lag_auto_cov(values, k, mean=None):
    if mean is None:
        mean = np.mean(values)
    return np.einsum('ij,ij->j',
                     (values[:-k] - mean), (values[k:] - mean)) / len(values)


def auto_cov(values, mean=None, variance=None):
    """ Compute the lag-autocovariance of a given array of values.

    :param values: Array of values.
    :param mean: Previously calculated or known mean.
    :param variance: Previously calculated or known variance.
    :return: Numpy array containing at index k the k-lag autocovariance.
    """
    size = values.shape[0]
    values = interpret_array(values)
    if mean is None:
        mean = np.mean(values, 0)
    if variance is None:
        variance = np.var(values, 0)
    centered = values - mean
    acov = np.empty_like(values)
    acov[0] = variance
    for k in range(1, size):
        acov[k] = np.einsum('ij,ij->j', centered[:-k], centered[k:]) / size
    return acov


def auto_corr(values, mean=None, variance=None):
    """ Compute the autocorrelation of a given array of values. """
    acov = auto_cov(values, mean, variance)
    return acov / acov[0]


def effective_sample_size(sample, mean, var):
    """ Estimate the effective sample size of a auto-correlated Markov sample.

    Estimated according to http://arxiv.org/abs/1111.4246

    :param sample: Sample object.
    :param mean: Mean of distribution, do not approximate via current sample!
    :param var: Variance of distribution, do not approximate via current sample!
    :return: Estimate of effective sample size.
    """
    mean = interpret_array(mean, sample.ndim)
    var = interpret_array(var, sample.ndim)
    sum = np.zeros(sample.ndim)

    unbias = sample.size / (sample.size - np.arange(sample.size))
    acor = auto_corr(sample.data, mean, var) * unbias[:, np.newaxis]
    for dim in range(sample.ndim):
        lag = 1
        rho = acor[lag, dim]
        while rho >= 0.05 and lag < sample.size - 1:
            sum[dim] = sum[dim] + (1 - lag / sample.size) * rho
            lag = lag + 1
            rho = acor[lag, dim]

    return sample.size / (1 + 2 * sum)


def fd_bins(sample):
    """ Estimates a good bin width. Use with caution.

    Use Freedman Diaconis Estimator with scaling like in sec 3.4 Eqn 3.61 (p83):
        Scott, D.W. (1992),
        Multivariate Density Estimation: Theory, Practice, and Visualization
    """
    mins = np.min(sample.data, axis=0)
    maxs = np.max(sample.data, axis=0)
    # h=2IQR(x)N^{−1/3} -> N^{-1/(2 + d)}
    widths = (2 * stats.iqr(sample.data.transpose(), axis=1) *
              sample.size ** (- 1 / (2 + sample.ndim)))
    bins = np.ceil((maxs - mins) / widths).astype(np.int)
    bins = np.maximum(1, bins)
    return np.asanyarray(bins).astype(np.int)


def bin_wise_chi2(sample, bins=None, bin_range=None,
                  min_count=10, int_steps=100):
    """ Compute the bin-wise chi^2 / dof value for a given number of bins.

    If the distribution of points in each bin follows a Poisson distribution
    with the expectation value according to the probability distribution,
    the returned value follows a chi squared distribution with expectation
    value 1.
    """
    try:
        pdf = sample.target.pdf
    except AttributeError:
        def pdf(xs):
            prob = np.empty(xs.shape[0])
            for j, x in zip(range(prob.size), xs):
                prob[i] = sample.target(x)
            return prob

    if bins is None:
        bins = fd_bins(sample)
    count, edges = np.histogramdd(sample.data, bins, bin_range)
    edges: tuple  # type hint for pycharm
    relevant = np.where(count >= min_count)

    expected = np.empty(len(relevant[0]))

    vol = np.prod([edges[d][1] - edges[d][0] for d in range(sample.ndim)])
    i = 0
    for mi in zip(*relevant):
        low = [edges[d][mi[d]] for d in range(sample.ndim)]
        high = [edges[d][mi[d]+1] for d in range(sample.ndim)]
        expected[i] = np.mean(pdf(np.random.uniform(
            low, high, (int_steps, sample.ndim)))) * sample.size * vol
        i += 1
    finals = np.where(expected >= min_count)[0]
    f_obs = count[relevant][finals]
    chi2, p = stats.chisquare(f_obs, f_exp=expected[finals])
    if len(f_obs > 0):
        return chi2 / (f_obs.size-1), p, f_obs.size
    return None, None, None
