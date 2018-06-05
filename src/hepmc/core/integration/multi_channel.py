import numpy as np
from matplotlib import pyplot as plt

from ..density import Distribution
from ..sampling import Sample


class ChannelsSample(Sample):

    def __init__(self, channel_weights, data, weights, sample_sizes, **kwargs):
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
        super().__init__(data=data, weights=weights, **kwargs)

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


class MultiChannel(Distribution):

    def __init__(self, channels, channels_weight=None):
        """ Channels construct for multi channel Monte Carlo.

        Contains several importance sampling channels (distributions and
        sampling methods) with respective weights and provides overall sampling
        and distribution functions.

        :param channels: List of distributions.
        :param channels_weight: Initial weight of the channels. By default
            assign equal weight to all channels.
        """
        ndim = channels[0].ndim
        for c in channels:
            if c.ndim != ndim:
                raise RuntimeError("Not all channels have the same ndim.")

        super().__init__(ndim)

        self.count = len(channels)
        if channels_weight is not None:
            self.channels_weight = channels_weight
        else:
            # equal weight to each channel
            self.channels_weight = np.ones(self.count) / self.count
        # keep the original channel weights to allow reset
        self.init_channel_weights = np.copy(self.channels_weight)

        self.channels = channels

        # later store information of generated sample
        self.current_sample = None  # ChannelSample

    def reset(self):
        """ Revert weights to initial values. """
        self.channels_weight[:] = self.init_channel_weights

    def pdf(self, xs):
        """  Overall probability of sample points x.

        Returns overall probability density of sampling a given point x:
        sum_i channel_weight_i * channel_pdf_i(x)

        :param xs: Total of self.ndim numpy array of equal lengths N.
        :return: Probabilities for each sample point. Numpy array
            of length N.
        """
        p = 0
        for i in range(self.count):
            p += self.channels_weight[i] * self.channels[i].pdf(xs)
        return p

    def pdf_gradient(self, xs):
        p = 0
        for i in range(self.count):
            p += self.channels_weight[i] * self.channels[i].pdf_gradient(xs)
        return p

    def plot_pdf(self, label="total pdf"):
        """ Plot the overall probability distribution. """
        x = np.linspace(0, 1, 1000)
        y = [self.pdf(xi) for xi in x]
        plt.plot(x, y, label=label)

    def rvs(self, sample_size, return_sizes=False):
        """ Generate a sample, using each channel with the given weight.

        To generate one point in the sample, a channel is chosen with
        the channel_weight as probability and using the corresponding sampling
        method.

        :param sample_size: Number of samples to generate.
        :param return_sizes: If true, count the number of samples generated
            using each channel.
        :return: Numpy array of shape (sample_size, self.ndim).
            If return_sizes is true, return a tuple of the sample and
            another numpy array specifying the number of samples generated
            using each channel.
        """
        sample_sizes = np.random.multinomial(sample_size, self.channels_weight)
        sample_channel_indices = np.where(sample_sizes > 0)[0]

        sample_points = np.concatenate(
            [self.channels[i].rvs(sample_sizes[i])
             for i in sample_channel_indices])

        if return_sizes:
            return sample_points, sample_sizes
        else:
            return sample_points

    def sample(self, sample_size):
        """ Generate a full sample and information for multi channel MC.

        The generated ChannelSample containing all information of the channel
        is returned and saved as self.current_sample.

        :param sample_size: Number of sample points.
        :return: A ChannelsSample object.
        """
        sample, sample_size = self.rvs(sample_size, True)
        weights = self.pdf(sample)
        self.current_sample = ChannelsSample(
            self.channels_weight, sample, weights, sample_size)
        return self.current_sample

    def update_channel_weights(self, new_channel_weights):
        """ Update the channel weights with given values.

        :param new_channel_weights: Channel weights of active channels.
            All remaining weights are set to zero.
        """
        self.channels_weight.fill(0)
        active_indices = self.current_sample.active_channels
        self.channels_weight[active_indices] = new_channel_weights
