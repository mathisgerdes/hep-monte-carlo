import numpy as np
from matplotlib import pyplot as plt

from ..density import Distribution
from ..util import interpret_array, hypercube_bounded


# VOLUMES for Stratified Sampling
# For the stratified monte carlo variant we first need a way to encode
# the volumes, then iterate over them and sample each one appropriately.
class GridVolumes(Distribution):

    def __init__(self, bounds=None, base_counts=None, default_base_count=1,
                 ndim=1, divisions=1):
        """ Grid-like partition of a the hypercube [0, 1]^ndim.

        Each partition has assigned a base count. This class provides methods
        to adapt the size of the division and sample points randomly for
        stratified sampling and VEGAS.

        :type divisions: int, iterable
        :param bounds: Tuple of lists; accumulative boundaries of the volumes.
            Example: bounds=([0, .5, 1]) for two 1D partitions of equal length.
        :param base_counts: Dictionary specifying base number of samples
            for bins. The key must indicate the multi-index (tuple!) of the bin.
        :param default_base_count: The default number of samples for bins not
            included in counts. Must be an integer value.
        :param ndim: Dimensionality of the volume. Ignored if bounds is given.
        :param divisions: Number of divisions along each dimension. Ignored
            if bounds is given.
        """
        if base_counts is None:
            # no entries, always use default_base_count
            base_counts = dict()

        self.base_counts = base_counts
        self.default_base_count = default_base_count

        # later assigned via setters
        self.total_base_count = None
        self.partition_count = None
        self._bounds = None
        self._sizes = None

        if bounds is None:
            super().__init__(ndim, False)
            try:
                if len(divisions) != ndim:
                    raise RuntimeError("Divisions must be scalar or of "
                                       "length ndim")
                self.bounds = [np.linspace(0, 1, divs+1) for divs in divisions]
            except TypeError:
                self.bounds = [np.linspace(0, 1, divisions + 1)
                               for _ in range(ndim)]

        else:
            super().__init__(len(bounds), False)
            self.bounds = bounds

        # allow bounds to be modified and later reset
        self.initial_bounds = [np.copy(b) for b in self.bounds]

    def reset(self):
        """ Reset bounds to initial values. """
        self.bounds = [np.copy(b) for b in self.initial_bounds]

    def pdf_indices(self, indices):
        vols = np.prod([self._sizes[d][indices[d]]
                        for d in range(self.ndim)], axis=0)

        counts = [self.get_count(index) for index in indices.transpose()]
        return np.array(counts) / self.total_base_count / vols

    @hypercube_bounded(1, self_has_ndim=True)
    def pdf(self, xs):
        xs = interpret_array(xs, self.ndim)
        # bin indices for all x values in each dimension
        indices = np.empty(xs.transpose().shape, dtype=np.int)
        for dim in range(self.ndim):
            indices[dim] = np.argmax(xs[:, dim, np.newaxis] <
                                     self.bounds[dim], axis=1) - 1

        return self.pdf_indices(indices)

    def pdf_gradient(self, xs):
        raise NotImplementedError("Density is a step function.")

    def plot_pdf(self, label="sampling weights"):
        """ Plot the effective probability density (esp. for VEGAS). """
        # visualization of 1d volumes
        assert self.ndim == 1, "Can only plot volumes in 1 dimension."
        # bar height corresponds to pdf
        height = [N / self.total_base_count / vol
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

    def sample_indices(self, indices):
        """ Note this returns a transposed sample array. """
        count = indices.shape[1]
        sample = np.empty((self.ndim, count))
        for d in range(self.ndim):
            sample[d] = (self._bounds[d][indices[d]] +
                         self._sizes[d][indices[d]] * np.random.rand(count))
        return sample

    def rvs(self, sample_count):
        indices = self.random_bins(sample_count)
        return self.sample_indices(indices).transpose()

    def get_count(self, index, multiple=1):
        # possible loss of "probability" (fewer effective samples)
        # if multiple is not an integer!
        index = tuple(index)
        if index in self.base_counts:
            return int(multiple * self.base_counts[index])
        else:
            return int(multiple * self.default_base_count)

    def iterate(self, multiple=1):
        lower = np.empty(self.ndim)
        upper = np.empty(self.ndim)
        for index in np.ndindex(*[len(b) - 1 for b in self.bounds]):
            count = self.get_count(index, multiple)
            for d in range(self.ndim):
                lower[d] = self.bounds[d][index[d]]
                upper[d] = self.bounds[d][index[d] + 1]
            samples = lower + (upper - lower) * np.random.rand(count, self.ndim)
            vol = np.prod(upper - lower)
            yield count, samples, vol

    def _update_total_base_count(self):
        partition_count = np.prod([s.size for s in self._sizes])
        total_count = sum(self.base_counts.values())
        total_count += self.default_base_count * (partition_count -
                                                  len(self.base_counts))
        self.total_base_count = total_count
        self.partition_count = partition_count

    @property
    def sizes(self):
        return self._sizes

    @sizes.setter
    def sizes(self, sizes):
        self._sizes = sizes
        # Set bounds according to partition sizes.
        for d in range(self.ndim):
            self.bounds[d] = np.empty(sizes[d].size + 1)
            self.bounds[d][1:] = np.cumsum(sizes[d])

        self._update_total_base_count()

    @property
    def bounds(self):
        return self._bounds

    @bounds.setter
    def bounds(self, bounds):
        if not isinstance(bounds, np.ndarray):
            bounds = [np.array(b, subok=True, copy=False) for b in bounds]
        self._bounds = bounds
        self._sizes = [np.diff(self.bounds[d]) for d in range(self.ndim)]

        self._update_total_base_count()
