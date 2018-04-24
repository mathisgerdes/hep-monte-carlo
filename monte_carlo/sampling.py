"""
Module containing several sampling methods and sampling structures used
in Monte Carlo integration and sampling techniques.

The core goal of sampling is to generate a sample of points distributed
according to a general (unnormalized) distribution function about which
little or nothing is known.
"""

import numpy as np
import matplotlib.pyplot as plt


def assure_2d(array):
    """ Assure the vector is two dimensional.

    Several of the sampling algorithms work with samples of shape (N, ndim),
    if ndim is one it is, however, convenient to simply have the shape (N,).
    This method adds an extra dimension if necessary to allow both shapes.

    If the array has more than 2 dimensions, an exception is raised.

    Example:
        >>> x = np.array([0, 1, 2, 3])
        >>> y = assure_2d(x)
        >>> y.shape
        (4, 1)
        >>> y is assure_2d(y)  # y is already two dimensional
        True

    :param array: Numpy array, either one or two dimensional.
    :return: If the original shape is (N,) the returned array has shape (N, 1).
        For arrays with shape (N, ndim) this function is the identity.
    """
    if array.ndim == 2:
        return array
    elif array.ndim == 1:
        return array[:, np.newaxis]
    else:
        raise RuntimeError("Array must be 1 or 2 dimensional.")


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


# CHANNELS for Monte Carlo Multi-Channel
class ChannelSample(object):
    def __init__(self, channel_weights, sample, sample_sizes, sample_weights):
        """ Store information about a sample generated using Channels.

        self.full_sample_sizes: Total number of points generated in each of
            the channels. This includes channels which did not contribute.
            In the remaining attributes, channels that do not contribute are
            ignored which is useful in processing the sample in multi channel
            Monte Carlo integration.

        self.active_channels: Indices of channels that contributed at least one
            of the sample points.

        self.channels_weight: Weights of the active channels.

        self.count_per_channel: Number of points sampled in each of the
            active channels.

        self.active_channel_count: Total number of active channels.

        self.channel_bounds: Starting index of sample points for each
            of the active channels in self.sample.
            Starts with zero and ends with a number smaller than the last
            index of self.sample

        self.sample: Generated sample, shape (sample_size, ndim) numpy array.

        self.sample_weights: Weights for each of the sample points.
            The weight is the probability density of the sample points.

        :param channel_weights: All channel weights used in generating
            the sample.
        :param sample: Generated sample.
        :param sample_sizes: Number of samples generated in each channel.
        :param sample_weights: Probability densities of the sample points.
        """
        self.full_sample_sizes = sample_sizes

        # ignore inactive channels i with sample_sizes[i] == 0
        self.active_channels = np.where(sample_sizes > 0)[0]
        self.channel_weights = channel_weights[self.active_channels]
        self.count_per_channel = sample_sizes[self.active_channels]

        # number of channels active in current sample
        self.active_channel_count = self.active_channels.size
        self.channel_bounds = np.array(
            [np.sum(self.count_per_channel[0:i])
             for i in range(self.active_channel_count)])

        self.sample = sample
        self.sample_weights = sample_weights


class Channels(object):
    def __init__(self, channels_sampling, channels_pdf, channels_weight=None):
        """ Channels construct for multi channel Monte Carlo.

        Contains several importance sampling channels (distributions and
        sampling methods) with respective weights and provides overall sampling
        and distribution functions.

        :param channels_sampling: Array of sampling functions for each of
            the channels.
        :param channels_pdf: Array of distributions for each of the sampling
            methods in sampling_channels.
        :param channels_weight: Initial weight of the channels. By default
            assign equal weight to all channels.
        """
        assert len(channels_sampling) == len(channels_pdf), \
            "Need a pdf for each sampling function."

        self.count = len(channels_sampling)
        if channels_weight is not None:
            self.channels_weight = channels_weight
        else:
            # equal weight to each channel
            self.channels_weight = np.ones(self.count) / self.count
        # keep the original channel weights to allow reset
        self.init_channel_weights = np.copy(self.channels_weight)
        self.sampling_channels = channels_sampling
        self.channel_pdfs = channels_pdf

        # later store information of generated sample
        self.current_sample = None  # ChannelSample

    def reset(self):
        """ Revert weights to initial values. """
        self.channels_weight[:] = self.init_channel_weights

    def pdf(self, *x):
        """  Overall probability of sample points x.

        Returns overall probability density of sampling a given point x:
        sum_i channel_weight_i * channel_pdf_i(x)

        :param x: Total of self.ndim numpy array of equal lengths N.
        :return: Probabilities for each sample point. Numpy array
            of length N.
        """
        p = 0
        for i in range(self.count):
            p += self.channels_weight[i] * self.channel_pdfs[i](*x)
        return p

    def plot_pdf(self, label="total pdf"):
        """ Plot the overall probability distribution. """
        x = np.linspace(0, 1, 1000)
        y = [self.pdf(xi) for xi in x]
        plt.plot(x, y, label=label)

    def sample(self, sample_size, return_sizes=False):
        """ Generate a sample, using each channel with the given weight.

        To generate one point in the sample, a channel is chosen with
        the channel_weight as probability and using the corresponding sampling
        method.

        :param sample_size: Number of samples to generate.
        :param return_sizes: If true, count the number of samples generated
            using each channel.
        :return: Numpy array of shape (sample_size, self.ndim).
            If return_sizes is true, return a tuple of the sample (numpy array)
            and a numpy array specifying the number of samples generated
            using each channel.
        """
        sample_sizes = np.random.multinomial(sample_size, self.channels_weight)
        sample_channel_indices = np.where(sample_sizes > 0)[0]

        sample_points = np.concatenate(
            [assure_2d(self.sampling_channels[i](sample_sizes[i]))
             for i in sample_channel_indices])

        if return_sizes:
            return sample_points, sample_sizes
        else:
            return sample_points

    def generate_sample(self, sample_size):
        """ Generate a full sample and information for multi channel MC.

        The generated ChannelSample containing all information of the channel
        is returned and saved as self.current_sample.

        :param sample_size: Number of sample points.
        :return: A ChannelSample object.
        """
        sample, sample_size = self.sample(sample_size, True)
        weights = self.pdf(*sample.transpose())
        self.current_sample = ChannelSample(self.channels_weight,
                                            sample, sample_size, weights)
        return self.current_sample

    def update_channel_weights(self, new_channel_weights):
        """ Update the channel weights with given values.

        :param new_channel_weights: Channel weights of active channels.
            All remaining weights are set to zero.
        """
        self.channels_weight.fill(0)
        active_indices = self.current_sample.active_channels
        self.channels_weight[active_indices] = new_channel_weights


# VOLUMES for Stratified Sampling
# For the stratified monte carlo variant we first need a way to encode
# the volumes, then iterate over them and sample each one appropriately.
class GridVolumes(object):
    def __init__(self, bounds=None, base_counts=None, default_count=1,
                 ndim=1, divisions=1):
        """ Grid-like partition of a the hypercube [0, 1]^ndim.

        Each partition has assigned a base count. This class provides methods
        to adapt the size of the division and sample points randomly for
        stratified sampling and VEGAS.

        :param bounds: Tuple of lists; accumulative boundaries of the volumes.
            Example: bounds=([0, .5, 1]) for two 1D partitions of equal length.
        :param base_counts: Dictionary specifying base number of samples
            for bins. The key must indicate the multi-index (tuple!) of the bin.
        :param default_count: The default number of samples for bins not
            included in counts. Must be an integer value.
        :param ndim: Dimensionality of the volume. Ignored if bounds is given.
        :param divisions: Number of divisions along each dimension. Ignored
            if bounds is given.
        """
        if base_counts is None:
            # no entries, always use default_base_count
            base_counts = dict()

        self.base_counts = base_counts
        self.default_base_count = default_count
        self.ndim = ndim
        self.total_base_size = sum(base_counts.values()) + \
            default_count * (divisions ** ndim - len(base_counts))
        if bounds is None:
            self.bounds = [np.linspace(0, 1, divisions + 1)
                           for _ in range(ndim)]
            self.ndim = ndim
        else:
            self.bounds = [np.array(b, subok=True, copy=False) for b in bounds]
            self.ndim = len(bounds)
        # allow bounds to be modified and later reset
        self.initial_bounds = [np.copy(b) for b in self.bounds]

    def reset(self):
        """ Reset bounds to initial values. """
        self.bounds = [np.copy(b) for b in self.initial_bounds]

    def plot_pdf(self, label="sampling weights"):
        """ Plot the effective probability density (esp. for VEGAS). """
        # visualization of 1d volumes
        assert self.ndim == 1, "Can only plot volumes in 1 dimension."
        # bar height corresponds to pdf
        height = [N / self.total_base_size / vol
                  for N, _, vol in self.iterate()]
        width = self.bounds[0][1:] - self.bounds[0][:-1]
        plt.bar(self.bounds[0][:-1], height, width, align='edge', alpha=.4,
                label=label)

    def update_bounds_from_sizes(self, sizes):
        """ Set bounds according to partition sizes. """
        for d in range(self.ndim):
            self.bounds[d][1:] = np.cumsum(sizes[d])

    def random_bins(self, count):
        """ Return the indices of N randomly chosen bins.

        :return: array of shape ndim x N of bin indices.
        """
        indices = np.empty((self.ndim, count), dtype=np.int)
        for d in range(self.ndim):
            indices[d] = np.random.randint(0, self.bounds[d].size - 1, count)

        return indices

    def sample(self, indices):
        """ Note this returns a transposed sample array. """
        count = indices.shape[1]
        sample = np.empty((self.ndim, count))
        for d in range(self.ndim):
            lower = self.bounds[d][indices[d]]
            upper = self.bounds[d][indices[d] + 1]
            sample[d] = lower + (upper - lower) * np.random.rand(count)
        return sample

    def iterate(self, multiple=1):
        lower = np.empty(self.ndim)
        upper = np.empty(self.ndim)
        for index in np.ndindex(*[len(b) - 1 for b in self.bounds]):
            if index in self.base_counts:
                count = int(multiple * self.base_counts[index])
            else:
                count = int(multiple * self.default_base_count)
            for d in range(self.ndim):
                lower[d] = self.bounds[d][index[d]]
                upper[d] = self.bounds[d][index[d] + 1]
            samples = lower + (upper - lower) * np.random.rand(count, self.ndim)
            vol = np.prod(upper - lower)
            yield count, samples, vol
