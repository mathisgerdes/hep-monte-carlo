import numpy as np
from ..density import Proposal


class UniformLocal(Proposal):

    def __init__(self, ndim, delta, sample_range=(0, 1)):
        super().__init__(ndim, True)

        self.low = sample_range[0]
        self.high = sample_range[1]
        self.vol = np.prod(np.diff(sample_range, axis=0))

        self._delta = None
        self.delta = delta

    def proposal(self, state):
        """ Uniformly generate a point in a self.delta environment.

        If state is close to the edges in a dimension, the uniform range
        is asymmetric such that the length is preserved but within the unit
        hypercube [0, 1]^ndim.

        :param state: The state (point) around which to sample.
        :return: Numpy array of length ndim.
        """
        base = np.maximum(self.low, state - self.delta / 2)
        base_capped = np.minimum(base, self.high - self.delta)
        return base_capped + np.random.rand() * self.delta

    def proposal_pdf(self, state, candidate):
        return 1 / np.prod(self.delta)  # symmetric

    @property
    def delta(self):
        return self._delta

    @delta.setter
    def delta(self, delta):
        if np.ndim(delta) == 0:
            self._delta = np.ones(self.ndim) * delta
        elif len(delta) == self.ndim:
            self._delta = np.array(delta)
        else:
            raise ValueError('delta must be a float or an array of '
                             'length ndim.')

        if np.any((delta - self.vol) > 0):
            raise ValueError('delta must be smaller than sample area.')
