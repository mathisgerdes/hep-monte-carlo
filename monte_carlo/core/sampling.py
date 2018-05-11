"""
Module containing several sampling methods and sampling structures used
in Monte Carlo integration and sampling techniques.

The core goal of sampling is to generate a sample of points distributed
according to a general (unnormalized) distribution function about which
little or nothing is known.
"""

import numpy as np
from .density import Density
from .util import interpret_array, effective_sample_size, bin_wise_chi2
from .sample_plotting import plot1d, plot2d


class Sample(object):
    def __init__(self, **kwargs):
        self._data = None
        self._target = None

        self.variance = None
        self.mean = None
        self.weights = None

        self.effective_sample_size = None
        self.bin_wise_chi2 = None

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def extend_array(self, key, array):
        current = getattr(self, key)
        if current is None:
            setattr(self, key, array)
        else:
            setattr(self, key, np.concatenate((current, array), axis=0))

    def extend_all(self, *args):
        if len(args) % 2 != 0:
            raise RuntimeError("Must pass alternating keys and values")

        for i in range(len(args) // 2):
            self.extend_array(args[i], args[i + 1])

    # PROPERTIES
    @property
    def size(self):
        try:
            return self.data.shape[0]
        except AttributeError:
            return None

    @property
    def ndim(self):
        try:
            return self.data.shape[1]
        except AttributeError:
            return None

    # PROPERTIES TRIGGERING UPDATES
    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.update_data()

    @property
    def target(self):
        return self._target

    @target.setter
    def target(self, target):
        self._target = Density.make(target, self.ndim)
        if self.data is not None:
            self.update_data_target()

    # UPDATES
    def update_data(self):
        self.mean = np.mean(self._data, axis=0)
        self.variance = np.var(self._data, axis=0)
        if self.target is not None:
            self.update_data_target()

    def update_data_target(self):
        self.bin_wise_chi2 = bin_wise_chi2(self)

        if self.ndim == 1:
            try:
                self.effective_sample_size = effective_sample_size(
                    self.data, self.target.mean, self.target.variance)
            except AttributeError:
                # target may not have known mean/variance
                pass

    def plot(self):
        if self.data is None:
            return None
        if self.ndim == 1:
            return plot1d(self, target=self.target)
        if self.ndim == 2:
            return plot2d(self, target=self.target)

    def _data_table(self):
        titles = ['Data (size)', 'mean', 'variance',
                  'effective sample size']
        entries = [self.size, self.mean, self.variance,
                   self.effective_sample_size]
        entries = [str(e) for e in entries]

        if self.bin_wise_chi2 is not None and self.bin_wise_chi2[0] is not None:
            entries.append('%.4g, p=%.4g' % self.bin_wise_chi2)
            titles.append('bin-wise chi^2')

        return titles, entries

    def _html_list(self):
        titles, entries = self._data_table()
        info = ['<h3>' + type(self).__name__ + '</h3>',
                '<table><tr><th style="text-align:left;">' +
                '</th><th style="text-align:left;">'.join(titles) +
                '</th></tr><tr><td style="text-align:left;">' +
                '</td><td style="text-align:left;">'.join(entries) +
                '</td></tr></table>']
        return info

    def _repr_html_(self):
        return '\n'.join(self._html_list())

    def _repr_png_(self):
        self.plot()

    def __repr__(self):
        return (type(self).__name__ + '\n\t' + '\n\t'.join(
            '%s: %s' % (t, e) for t, e in zip(*self._data_table())))


# ACCEPTANCE REJECTION
class AcceptRejectSampler(object):

    def __init__(self, pdf, bound, ndim=1, sampling=None, sampling_pdf=None):
        """ Acceptance Rejection method for sampling a given pdf.

        The method uses a known distribution and sampling method to propose
        samples which are then accepted with the probability
        pdf(x)/(c * sampling_pdf(x)), thus producing the desired distribution.

        :param pdf: Unnormalized desired probability distribution of the sample.
        :param bound: Constant such that pdf(x) <= bound * sampling_pdf(x)
            for all x in the range of sampling.
        :param ndim: Dimensionality of the sample points.
            This must conform with sampling and sampling_pdf.
        :param sampling: Generate a given number of samples according to
            sampling_pdf. The default is a uniform distribution. The algorithm
            gets more efficient, the closer the sampling is to the desired
            distribution pdf(x).
        :param sampling_pdf: Returns the probability of sampling to generate
            a given sample. Must accept ndim arguments, each of some
            length N and return an array of floats of length N. Ignored if
            sampling was not specified.
        """
        self.pdf = pdf
        self.c = bound
        self.ndim = ndim

        if sampling is None:
            def sampling(sample_size):
                """ Generate a uniform sample. """
                sample = np.random.rand(sample_size * self.ndim)
                return sample.reshape(sample_size, self.ndim)

            def sampling_pdf(*_):
                """ Uniform sample distribution. """
                return 1

        self.sampling = sampling
        self.sampling_pdf = sampling_pdf

    def sample(self, sample_size, statistics=True):
        """ Generate a sample according to self.pdf of given size.

        :param sample_size: Number of samples
        :return: Numpy array with shape (sample_size, self.ndim).
        """
        x = np.empty((sample_size, self.ndim))

        indices = np.arange(sample_size)
        while indices.size > 0:
            proposal = interpret_array(self.sampling(indices.size), self.ndim)
            accept = np.random.rand(indices.size) * self.c * self.sampling_pdf(
                *proposal.transpose()) <= self.pdf(*proposal.transpose())
            x[indices[accept]] = proposal[accept]
            indices = indices[np.logical_not(accept)]
        if statistics:
            return Sample(data=x, target=self.pdf)
        else:
            return Sample(data=x)
