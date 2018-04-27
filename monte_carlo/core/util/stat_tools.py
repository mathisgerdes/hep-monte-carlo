"""
Module provides methods for analyzing the statistics of a sample
(or values) generated with Monte Carlo techniques.
"""

import numpy as np


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
        acov[k] = np.sum((values[:-k] - mean) * (values[k:] - mean)) / size
    return acov


def auto_corr(values):
    """ Compute the autocorrelation of a given array of values. """
    acov = auto_cov(values)
    return acov / acov[0]


def binwise_chi2(pdf, values, bins=400, pdf_range=None):
    """ Compute the bin-wise chi^2 / dof value for a given number of bins.

    If the distribution of points in each bin follows a Poisson distribution
    with the expectation value according to the probability distribution,
    the returned value follows a chi squared distribution with expectation
    value 1.

    :param pdf: A function giving the probability density for the distribution.
    :param values: An array of values that is supposed to follow pdf.
    :param bins: Number of bins to count the number of points in.
    :param pdf_range: Tuple, range over which to compare the value distribution
        to the pdf. If None it defaults to the maximal and minimal entries
        in the values array.
    :return: The chi^2 / dof value.
    """
    if pdf_range is None:
        pdf_range = (np.min(values), np.max(values))

    total = len(values)
    bin_borders = np.linspace(*pdf_range, bins + 1)
    bin_width = bin_borders[1] - bin_borders[0]
    # number of samples in each bin
    counts = np.array([np.count_nonzero((values < bin_borders[i + 1]) *
                                        (values > bin_borders[i]))
                       for i in range(bins)])
    predicted_counts = np.array([total * bin_width * (pdf(bin_borders[i + 1]) +
                                                      pdf(bin_borders[i])) / 2
                                 for i in range(bins)])

    chi2 = np.sum((counts - predicted_counts) ** 2 / predicted_counts)
    return chi2 / (bins - 1)  # dof = bins - 1
