import numpy as np
from matplotlib import pyplot as plt


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
