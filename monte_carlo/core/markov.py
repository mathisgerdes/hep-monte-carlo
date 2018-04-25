import numpy as np


# MARKOV CHAIN
class AbstractMarkovUpdate(object):

    """ Basic update mechanism of a Markov chain. """
    def __init__(self, ndim):
        self.ndim = ndim

        # initialize these values with init_sampler, if needed.
        self.state = None
        self.out_mask = None

    def init_sampler(self, initial, out_mask=None):
        """ Initialize a sampler given the update this object specifies.

        :param initial: Initial value of the Markov chain. Internally
            converted to numpy array.
        :param out_mask: Slice object, return only this slice of the output
            chain (useful if sampler uses artificial variables).
        """
        initial = np.array(initial, copy=False, subok=True, ndmin=1)
        if len(initial) != self.ndim:
            raise ValueError("initial must be of dimension self.ndim=" +
                             str(self.ndim))
        self.state = initial
        self.out_mask = out_mask

    def next_state(self, state):
        """ Get the next state in the Markov chain.

        Depends on self.state, but must not change it.

        :return: The next state.
        """
        raise NotImplementedError("AbstractMarkovUpdate is abstract.")

    def sample(self, sample_size=1, return_accept_rate=False, log=5000):
        """ Generate a sample of given size.

        :param sample_size: Number of samples to generate.
        :param return_accept_rate: If true, compute the acceptance rate.
        :param log: Print the number of generated samples. Do not log if
            value is < 0. Log every sample for log=1.
        :return: Numpy array with shape (sample_size, self.ndim).
            If get_accept_rate is true, return a tuple of the array and
            the acceptance rate of the Metropolis algorithm in this run.
        """
        # check if sampler was initialized
        if self.state is None:
            raise RuntimeError("Call init_sampler before sampling according "
                               "to a Markov update.")

        chain = np.empty((sample_size, self.ndim))

        # only used if return_accept_rate is true.
        accepted = 0

        for i in range(sample_size):
            next_state = self.next_state(self.state)
            if return_accept_rate and not np.array_equal(next_state, self.state):
                accepted += 1
            chain[i] = self.state = next_state
            if log > 0 and (i+1) % log == 0:
                print("Generated %d samples." % (i+1))

        if self.out_mask is not None:
            chain = chain[:, self.out_mask]

        if return_accept_rate:
            return chain, accepted / sample_size
        return chain


# METROPOLIS (HASTING) UPDATES
class MetropolisLikeUpdate(AbstractMarkovUpdate):
    """ Generic abstract class to represent a single Metropolis-like update.
    """

    def accept(self, state, candidate):
        """ This function must be implemented by child classes.

        :param state: Previous state in the Markov chain.
        :param candidate: Candidate for next state.
        :return: The acceptance probability of a candidate state given the
            previous state in the Markov chain.
        """
        raise NotImplementedError("MetropolisLikeUpdate is abstract.")

    def proposal(self, state):
        """ A proposal generator.

        Generate candidate points in the sample space.
        These are used in the update mechanism and
        accepted with a probability self.accept(candidate) that depends
        on the used algorithm.

        :param state: The previous state in the Markov chain.
        :return: A candidate state.
        """
        raise NotImplementedError("MetropolisLikeUpdate is abstract.")

    def next_state(self, state):
        candidate = self.proposal(state)
        accept_prob = self.accept(state, candidate)
        if accept_prob >= 1 or np.random.rand() < accept_prob:
            return candidate
        return state


class MetropolisUpdate(MetropolisLikeUpdate):

    def __init__(self, ndim, target_pdf, proposal=None, proposal_pdf=None):
        """ Use the Metropolis algorithm to generate a sample.

        The dimensionality of the sample points is inferred from the length
        of initial.

        The proposal must not depend on the current state, if proposal_pdf
        is None (in that case simple Metropolis is used). If proposal is
        None, use Metropolis with a uniform proposal in [0,1].

        Example:
            >>> pdf = lambda x: np.sin(10*x)**2
            >>> met = MetropolisUpdate(1, pdf)   # 1 dimensional
            >>> met.init_sampler(0.1)            # initialize with start value
            >>> sample = met.sample(1000)        # generate 1000 samples


        :param target_pdf: Desired (unnormalized) probability distribution.
        :param proposal: A proposal generator.
            Takes the previous state as argument, but for this algorithm
            to work must not depend on it (argument exists only for generality
            of the implementation.)
        """
        super().__init__(ndim)

        if proposal is None:
            def proposal(_):
                """ Uniform proposal generator. """
                return np.random.rand(self.ndim)

        self.is_hasting = proposal_pdf is not None
        self._proposal = proposal
        self._proposal_pdf = proposal_pdf
        self._target_pdf = target_pdf

    def accept(self, state, candidate):
        """ Probability of accepting candidate as next state. """
        if self.is_hasting:
            return (self._target_pdf(candidate) *
                    self._proposal_pdf(candidate, state) /
                    self._target_pdf(state) *
                    self._proposal_pdf(state, candidate))

        # otherwise Metropolis update
        return self._target_pdf(candidate) / self._target_pdf(state)

    def proposal(self, state):
        """ Propose a candidate state. """
        return self._proposal(state)


class CompositeMarkovUpdate(AbstractMarkovUpdate):

    def __init__(self, ndim, updates, masks=None):
        """ Composite Metropolis sampler; combine updates.

        :param updates: List of update mechanisms, each subtypes of
            MetropolisLikeUpdate.
        :param masks: Dictionary, giving masks (list/array of indices)
            of dimensions for the index of the update mechanism. Use this if
            some updates only affect slices of the state.
        """
        super().__init__(ndim)
        self.updates = updates
        self.masks = [None if masks is None or i not in masks else masks[i]
                      for i in range(len(updates))]

    def next_state(self, state):
        for mechanism, mask in zip(self.updates, self.masks):
            if mask is None:
                state = mechanism.next_state(state)
            else:
                state = np.copy(state)
                state[mask] = mechanism.next_state(state[mask])

        return state


class MixingMarkovUpdate(AbstractMarkovUpdate):

    def __init__(self, ndim, updates, weights=None, masks=None):
        """ Mix a number of update mechanisms, choosing one in each step.

        :param updates: List of update mechanisms (AbstractMarkovUpdate).
        :param weights: List of weights for each of the mechanisms (sum to 1).
        :param masks: Slice object, specify if updates only affect slice of
            state.
        """
        super().__init__(ndim)
        self.updates = updates
        self.updates_count = len(updates)
        self.masks = [None if masks is None or i not in masks else masks[i]
                      for i in range(len(updates))]
        if weights is None:
            weights = np.ones(self.updates_count) / self.updates_count
        self.weights = weights

    def next_state(self, state):
        index = np.random.choice(self.updates_count, p=self.weights)
        if self.masks[index] is None:
            return self.updates[index].next_state(state)
        else:
            mask = self.masks[index]
            state = np.copy(state)
            state[mask] = self.updates[index].next_state(state[mask])
            return state
