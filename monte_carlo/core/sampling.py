"""
Module containing several sampling methods and sampling structures used
in Monte Carlo integration and sampling techniques.

The core goal of sampling is to generate a sample of points distributed
according to a general (unnormalized) distribution function about which
little or nothing is known.
"""

import numpy as np
from .util import interpret_array


class Sample(object):
    def __init__(self, **kwargs):
        self._data = None

        self.variance = None
        self.mean = None
        self.weights = None

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
            self.extend_array(args[i], args[i+1])

    @property
    def size(self):
        return self.data.shape[0]

    @property
    def ndim(self):
        return self.data.shape[1]

    @property
    def data(self):
        return self._data

    @data.setter
    def data(self, data):
        self._data = data
        self.mean = np.mean(data, axis=0)
        self.variance = np.var(data, axis=0)


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
        return Sample(data=x)
