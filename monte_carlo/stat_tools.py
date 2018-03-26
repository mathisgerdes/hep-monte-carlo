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


def binwise_chi2(pdf, Xi, bins=400, pdf_range=None):
    """ Compute the bin-wise chi^2 / dof value for a given number of bins.

    Args:
        pdf: a function giving the probability density for the distribution.
        Xi: the sample that is supposed to follow pdf.
        bins: number of bins to count the number of points in.
        pdf_range: tuple, range over which to compare the distribution to pdf.
            If None it defaults to maximal and minimal values of Xi.
    """
    if pdf_range is None:
        pdf_range =(np.min(Xi), np.max(Xi))
    total = len(Xi)
    bin_borders = np.linspace(*pdf_range, bins+1)
    bin_width = bin_borders[1]- bin_borders[0]
    bin_counts = np.array([np.count_nonzero((Xi<bin_borders[i+1])*(Xi>bin_borders[i]))
                           for i in range(bins)])
    bin_counts_pred = np.array([total*bin_width*(pdf(bin_borders[i+1])+pdf(bin_borders[i]))/2
                                for i in range(bins)])

    return np.sum((bin_counts-bin_counts_pred)**2 / (bin_counts_pred)) / (bins-1)
