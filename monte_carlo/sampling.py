import numpy as np
import matplotlib.pyplot as plt


def assure_2d(array):
    """ Assure the vector is two dimensional.

    Several of the sampling algorithms work with samples of shape (N, dim),
    if dim is one it is, however, convenient to simply have the shape (N,).
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
        For arrays with shape (N, dim) this function is the identity.
    """
    if array.ndim == 2:
        return array
    elif array.ndim == 1:
        return array[:, np.newaxis]
    else:
        raise RuntimeError("Array must be 1 or 2 dimensional.")


# ACCEPTANCE REJECTION
class AcceptReject(object):

    def __init__(self, pdf, c, dim=1, sampling_pdf=lambda *x: 1, sampling=None):
        """ Acceptance Rejection method for sampling a given pdf.

        The method uses a known distribution and sampling method to propose
        samples which are then accepted with the probability
        pdf(x)/(c * sampling_pdf(x)), thus producing the desired distribution.

        :param pdf: Unnormalized desired probability distribution of the sample.
        :param c: Constant such that pdf(x) <= c * sampling_pdf(x) everywhere.
        :param dim: Dimensionality of the sample points.
            This must conform with sampling and sampling_pdf.
        :param sampling_pdf: Returns the probability of sampling to generate
            a given sample. Must accept dim arguments, each of some
            length N and return an array of floats of length N.
        :param sampling: Generate a given number of samples according to
            sampling_pdf. The default is a uniform distribution. The algorithm
            gets more efficient, the closer the sampling is to the desired
            distribution pdf(x).
        """
        self.pdf = pdf
        self.c = c
        self.dim = dim
        self.sample_pdf = sampling_pdf

        if sampling is None:
            def sampling(sample_size):
                """ Generate a uniform sample. """
                sample = np.random.rand(sample_size * self.dim)
                return sample.reshape(sample_size, self.dim)

        self.sample = sampling

    def __call__(self, sample_size):
        """ Generate a sample according to self.pdf of given size.

        :param sample_size: Number of samples
        :return: Numpy array with shape (sample_size, self.dim).
        """
        x = np.empty((sample_size, self.dim))

        indices = np.arange(sample_size)
        while indices.size > 0:
            proposal = assure_2d(self.sample(indices.size))
            accept = np.random.rand(indices.size) * self.c * self.sample_pdf(
                *proposal.transpose()) <= self.pdf(*proposal.transpose())
            x[indices[accept]] = proposal[accept]
            indices = indices[np.logical_not(accept)]
        return x


# METROPOLIS MARKOV CHAINS
class GenericMetropolis(object):
    def __init__(self, initial, pdf, dim, proposal):
        """ Generic Metropolis (Hasting) sampler.

        Class is abstract, child class has to implement a function 'accept'.
        Function, takes the previous and next state and returns
        the acceptance probability. (Values greater than 1 are treated as 1.)

        :param initial: Initial value of the Markov chain.
        :param pdf: Desired (unnormalized) probability distribution.
        :param dim: Dimensionality of points in the sample.
        :param proposal: A proposal generator (generate candidate points in
            the sample space. These are used in the update mechanism and
            accepted with a probability self.accept(candidate) that depends
            on the used algorithm. Takes the previous state as argument.
        """
        self.state = initial
        self.pdf = pdf
        self.dim = dim
        self.proposal = proposal

    def accept(self, state, candidate):
        """ This function must be implemented by child classes.

        :param state: Previous state in the Markov chain.
        :param candidate: Candidate for next state.
        :return: The acceptance probability of a candidate state given the
            previous state in the Markov chain.
        """
        raise NotImplementedError("GenericMetropolis is abstract.")

    def __call__(self, sample_size=1, get_accept_rate=False):
        """ Generate a sample of given size.

        :param sample_size: Number of samples to generate.
        :param get_accept_rate: If true, compute the acceptance rate.
        :return: Numpy array with shape (sample_size, self.dim).
            If get_accept_rate is true, return a tuple of the array and
            the acceptance rate of the Metropolis algorithm in this run.
        """
        chain = np.empty((sample_size, self.dim))

        # only used if get_accept_rate is true.
        accepted = 0

        for i in range(sample_size):
            candidate = self.proposal(self.state)
            accept_prob = self.accept(self.state, candidate)
            if accept_prob >= 1 or np.random.rand() < accept_prob:
                self.state = chain[i] = candidate

                # The advantage of not checking in every integration is
                # negligible. This way is cleaner as it avoids code duplication.
                if get_accept_rate:
                    accepted += 1
            else:
                chain[i] = self.state

        if get_accept_rate:
            return chain, accepted / sample_size
        return chain


class Metropolis(GenericMetropolis):
    def __init__(self, initial, pdf, dim=1, proposal=None):
        """ Use the Metropolis algorithm to generate a sample.

        The proposal must not depend on the current state.
        Use the Metropolis Hasting algorithm if it does.

        :param initial: Initial value of the Markov chain.
        :param pdf: Desired (unnormalized) probability distribution.
        :param dim: Dimensionality of points in the sample.
        :param proposal: A proposal generator.
            Takes the previous state as argument, but for this algorithm
            to work must not depend on it (argument exists only for generality
            of the implementation.)
        """
        if proposal is None:
            # default to uniform proposal distribution
            def proposal(_):
                """ Generate a sample with uniform distribution. """
                return np.random.rand(dim)

        super().__init__(initial, pdf, dim, proposal)

    def accept(self, state, candidate):
        """ Probability of accepting candidate as next state. """
        return self.pdf(candidate) / self.pdf(state)


class MetropolisHasting(GenericMetropolis):
    def __init__(self, initial, pdf, dim=1, proposal_pdf=None, proposal=None):
        """ Metropolis Hasting sampler.

        If proposal_pdf and proposal are not specified (None),
        a uniform distribution is used. This makes the method equivalent to
        the simpler Metropolis algorithm as the candidate distribution does
        not depend on the previous state.

        proposal_pdf and proposal must either both be specified or both None.

        :param initial: Initial value of the Markov chain.
        :param pdf: Desired (unnormalized) probability distribution.
        :param dim: Dimensionality of points in the sample.
        :param proposal: Function that generates a candidate state, given
            the previous state as single argument.
        :param proposal_pdf: Distribution of a proposed candidate.
            Takes the previous state and the generated candidate as arguments.
        """
        if proposal_pdf is None and proposal is None:
            # default to uniform proposal distribution
            def proposal_pdf(*_):  # candidate and state are not used.
                """ Uniform proposal distribution. """
                return 1

            def proposal(_):  # argument state is not used.
                """ Uniform candidate generation. """
                return np.random.rand(dim)
        elif proposal_pdf is None or proposal is None:
            raise ValueError("Cannot infer either proposal or proposal_pdf "
                             "from the other. Specify both or neither.")

        self.proposal_pdf = proposal_pdf
        super().__init__(initial, pdf, dim, proposal)

    def accept(self, state, candidate):
        """ Probability of accepting candidate as next state. """
        return (self.pdf(candidate) * self.proposal_pdf(candidate, state) /
                self.pdf(state) / self.proposal_pdf(state, candidate))


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

        self.channel_weights: Weights of the active channels.

        self.count_per_channel: Number of points sampled in each of the
            active channels.

        self.active_channel_count: Total number of active channels.

        self.channel_bounds: Starting index of sample points for each
            of the active channels in self.sample.
            Starts with zero and ends with a number smaller than the last
            index of self.sample

        self.sample: Generated sample, shape (sample_size, dim) numpy array.

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
    def __init__(self, sampling_channels, channel_pdfs, channel_weights=None):
        """ Channels construct for multi channel Monte Carlo.

        Contains several importance sampling channels (distributions and
        sampling methods) with respective weights and provides overall sampling
        and distribution functions.

        :param sampling_channels: Array of sampling functions for each of
            the channels.
        :param channel_pdfs: Array of distributions for each of the sampling
            methods in sampling_channels.
        :param channel_weights: Initial weight of the channels. By default
            assign equal weight to all channels.
        """
        assert len(sampling_channels) == len(channel_pdfs), \
            "Need a pdf for each sampling function."

        self.count = len(sampling_channels)
        if channel_weights is not None:
            self.channel_weights = channel_weights
        else:
            # equal weight to each channel
            self.channel_weights = np.ones(self.count) / self.count
        # keep the original channel weights to allow reset
        self.init_channel_weights = np.copy(self.channel_weights)
        self.sampling_channels = sampling_channels
        self.channel_pdfs = channel_pdfs

        # later store information of generated sample
        self.current_sample = None  # ChannelSample

    def reset(self):
        """ Revert weights to initial values. """
        self.channel_weights[:] = self.init_channel_weights

    def pdf(self, *x):
        """  Overall probability of sample points x.

        Returns overall probability density of sampling a given point x:
        sum_i channel_weight_i * channel_pdf_i(x)

        :param x: Total of self.dim numpy array of equal lengths N.
        :return: Probabilities for each sample point. Numpy array
            of length N.
        """
        p = 0
        for i in range(self.count):
            p += self.channel_weights[i] * self.channel_pdfs[i](*x)
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
        :return: Numpy array of shape (sample_size, self.dim).
            If return_sizes is true, return a tuple of the sample (numpy array)
            and a numpy array specifying the number of samples generated
            using each channel.
        """
        choice = np.random.rand(sample_size)
        sample_sizes = np.empty(self.count, dtype=np.int)
        cum_weight = 0  # cumulative weight of channels up to a certain index.
        for i in range(self.count):
            # choose a channel for each sample point by distributing the
            # number of samples over the channels using the respective weights.
            sample_sizes[i] = np.count_nonzero(
                np.logical_and(cum_weight < choice,
                               choice <= cum_weight + self.channel_weights[i]))
            cum_weight += self.channel_weights[i]

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
        self.current_sample = ChannelSample(self.channel_weights,
                                            sample, sample_size, weights)
        return self.current_sample

    def update_channel_weights(self, new_channel_weights):
        """ Update the channel weights with given values.

        :param new_channel_weights: Channel weights of active channels.
            All remaining weights are set to zero.
        """
        self.channel_weights.fill(0)
        active_indices = self.current_sample.active_channels
        self.channel_weights[active_indices] = new_channel_weights


# VOLUMES for Stratified Sampling
# For the stratified monte carlo variant we first need a way to encode
# the volumes, then iterate over them and sample each one appropriately.
class GridVolumes(object):
    def __init__(self, bounds=None, counts=None, default_count=1,
                 dim=1, divisions=1):
        """ Grid-like partition of a the hypercube [0, 1]^dim.

        Each partition has assigned a base count. This class provides methods
        to adapt the size of the division and sample points randomly for
        stratified sampling and VEGAS.

        :param bounds: Tuple of lists; accumulative boundaries of the volumes.
            Example: bounds=([0, .5, 1]) for two 1D partitions of equal length.
        :param counts: Dictionary specifying base number of samples for bins.
            The key must indicate the multi-index (tuple!) of the bin.
        :param default_count: The default number of samples for bins not
            included in counts. Must be an integer value.
        :param dim: Dimensionality of the volume. Ignored if bounds is given.
        :param divisions: Number of divisions along each dimension. Ignored
            if bounds is given.
        """
        if counts is None:
            # no entries, always use default_count
            counts = dict()

        self.Ns = counts
        self.otherNs = default_count
        self.dim = dim
        self.total_base_size = sum(counts.values()) + \
            (divisions ** dim - len(counts)) * default_count
        if bounds is None:
            self.bounds = [np.linspace(0, 1, divisions + 1) for _ in range(dim)]
            self.dim = dim
        else:
            self.bounds = [np.array(b) for b in bounds]
            self.dim = len(bounds)
        # allow bounds to be modified and later reset
        self.initial_bounds = [np.copy(b) for b in self.bounds]

    def reset(self):
        """ Reset bounds to initial values. """
        self.bounds = [np.copy(b) for b in self.initial_bounds]

    def plot_pdf(self, label="sampling weights"):
        """ Plot the effective probability density (esp. for VEGAS). """
        # visualization of 1d volumes
        assert self.dim == 1, "Can only plot volumes in 1 dimension."
        # bar height corresponds to pdf
        height = [N / self.total_base_size / vol
                  for N, _, vol in self.iterate()]
        width = self.bounds[0][1:] - self.bounds[0][:-1]
        plt.bar(self.bounds[0][:-1], height, width, align='edge', alpha=.4,
                label=label)

    def update_bounds_from_sizes(self, sizes):
        """ Set bounds according to partition sizes. """
        for d in range(self.dim):
            self.bounds[d][1:] = np.cumsum(sizes[d])

    def random_bins(self, count):
        """ Return the indices of N randomly chosen bins.

        :return: array of shape dim x N of bin indices.
        """
        indices = np.empty((self.dim, count), dtype=np.int)
        for d in range(self.dim):
            indices[d] = np.random.randint(0, self.bounds[d].size - 1, count)

        return indices

    def sample(self, indices):
        """ Note this returns a transposed sample array. """
        count = indices.shape[1]
        sample = np.empty((self.dim, count))
        for d in range(self.dim):
            lower = self.bounds[d][indices[d]]
            upper = self.bounds[d][indices[d] + 1]
            sample[d] = lower + (upper - lower) * np.random.rand(count)
        return sample

    def iterate(self, multiple=1):
        lower = np.empty(self.dim)
        upper = np.empty(self.dim)
        for index in np.ndindex(*[len(b) - 1 for b in self.bounds]):
            if index in self.Ns:
                count = int(multiple * self.Ns[index])
            else:
                count = int(multiple * self.otherNs)
            for d in range(self.dim):
                lower[d] = self.bounds[d][index[d]]
                upper[d] = self.bounds[d][index[d] + 1]
            samples = lower + (upper - lower) * np.random.rand(count, self.dim)
            vol = np.prod(upper - lower)
            yield count, samples, vol
