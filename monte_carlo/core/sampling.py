"""
Module containing several sampling methods and sampling structures used
in Monte Carlo integration and sampling techniques.

The core goal of sampling is to generate a sample of points distributed
according to a general (unnormalized) distribution function about which
little or nothing is known.
"""

import numpy as np
from .util import interpret_array, effective_sample_size, bin_wise_chi2
from .sample_plotting import plot1d, plot2d

import os
import time
import json


def _set(*args):
    return all(arg is not None for arg in args)


class Sample(object):
    def __init__(self, **kwargs):
        self.data = None
        self.target = None
        self.weights = None

        self._bin_wise_chi2 = None
        self._effective_sample_size = None
        self._variance = None
        self._mean = None

        self._sample_info = [
            ('size', 'data (size)', '%s'),
            ('mean', 'mean', '%s'),
            ('variance', 'variance', '%s'),
            ('bin_wise_chi2', 'bin-wise chi^2', '%.4g, p=%.4g, N=%d'),
            ('effective_sample_size', 'effective sample size', '%s'),
        ]

        for key in kwargs:
            setattr(self, key, kwargs[key])

    def extend_array(self, key, array):
        current = getattr(self, key)
        if current is None:
            setattr(self, key, array)
        else:
            setattr(self, key, np.concatenate((current, array), axis=0))

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

    @property
    def mean(self):
        if self._mean is None and _set(self.data):
            self._mean = np.mean(self.data, axis=0)
        return self._mean

    @property
    def variance(self):
        if self._variance is None and _set(self.data):
            self._variance = np.var(self.data, axis=0)
        return self._variance

    @property
    def effective_sample_size(self):
        if self._effective_sample_size is None and _set(self.data, self.target):
            try:
                self._effective_sample_size = effective_sample_size(
                    self, self.target.mean, self.target.variance)
            except AttributeError:
                # target may not have known mean/variance
                pass
        return self._effective_sample_size

    @property
    def bin_wise_chi2(self):
        if self._bin_wise_chi2 is None and _set(self.data, self.target):
            self._bin_wise_chi2 = bin_wise_chi2(self)
        return self._bin_wise_chi2

    def plot(self):
        if self.data is None:
            return None
        if self.ndim == 1:
            return plot1d(self, target=self.target)
        if self.ndim == 2:
            return plot2d(self, target=self.target)

    def save(self, file_path=None):
        if file_path is None:
            file_path = type(self).__name__ + '-' + str(int(time.time()))
        path, name = os.path.split(file_path)

        info = {entry[0]: repr(getattr(self, entry[0]))
                for entry in self._sample_info}
        info['type'] = type(self).__name__
        info['target'] = repr(self.target)

        np.save(os.path.join(path, name + '-data.npy'), self.data)
        with open(os.path.join(path, name + '.json'), 'w') as fp:
            json.dump(info, fp, indent=2)

    def _data_table(self):
        titles = [entry[1] for entry in self._sample_info]
        entries = []
        for entry in self._sample_info:
            value = getattr(self, entry[0])
            try:
                entries.append(entry[2] % value)
            except TypeError:
                entries.append('N/A')

        return titles, entries

    def _repr_html_(self):
        titles, entries = self._data_table()
        info = ['<h3>' + type(self).__name__ + '</h3>',
                '<table><tr><th style="text-align:left;">' +
                '</th><th style="text-align:left;">'.join(titles) +
                '</th></tr><tr><td style="text-align:left;">' +
                '</td><td style="text-align:left;">'.join(entries) +
                '</td></tr></table>']
        return '\n'.join(info)

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

    def sample(self, sample_size):
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
        return Sample(data=x, target=self.pdf)
