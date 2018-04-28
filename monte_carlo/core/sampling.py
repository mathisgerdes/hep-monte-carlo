"""
Module containing several sampling methods and sampling structures used
in Monte Carlo integration and sampling techniques.

The core goal of sampling is to generate a sample of points distributed
according to a general (unnormalized) distribution function about which
little or nothing is known.
"""

import numpy as np

from .util import assure_2d


class SampleInfo(object):

    def __init__(self):
        self.ndim = 0
        self.size = 0
        self.var = None
        self.mean = None
        self.accepted = 0


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
            proposal = assure_2d(self.sampling(indices.size))
            accept = np.random.rand(indices.size) * self.c * self.sampling_pdf(
                *proposal.transpose()) <= self.pdf(*proposal.transpose())
            x[indices[accept]] = proposal[accept]
            indices = indices[np.logical_not(accept)]
        return x
