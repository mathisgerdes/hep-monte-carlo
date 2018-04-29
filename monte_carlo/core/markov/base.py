import numpy as np
from ..sampling import SampleInfo


class MetropolisState(np.ndarray):
    def __new__(cls, input_array, pdf=None):
        obj = np.array(input_array, copy=False, subok=True, ndmin=1).view(cls)
        obj.pdf = pdf
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return  # was called from the __new__ above
        self.pdf = getattr(obj, 'pdf', None)


# MARKOV CHAIN
class AbstractMarkovUpdate(object):
    """ Basic update mechanism of a Markov chain. """

    def __init__(self, ndim, is_adaptive=False):
        self.ndim = ndim
        self.is_adaptive = is_adaptive

        # initialize these values with init_sampler, if needed.
        self.state = None
        self.out_mask = None
        self.get_info = False
        self.log_every = -1
        self.re_sample = None

        # will hold information if update was used as a sampler
        self.sample_info = None

    def init_adapt(self, initial_state):
        pass

    def init_state(self, state):
        return state  # may initialize other state attributes (such as pdf)

    def init_sampler(self, initial, out_mask=None,
                     get_info=False, log_every=5000):
        """ Initialize a sampler given the update this object specifies.

        :param initial: Initial value of the Markov chain. Internally
            converted to numpy array.
        :param out_mask: Slice object, return only this slice of the output
            chain (useful if sampler uses artificial variables).
        :param get_info: If true, compute the acceptance rate.
        :param log_every: Print the number of generated samples. Do not log if
            value is < 0. Log every sample for log=1.
        """
        self.state = self.init_state(np.atleast_1d(initial))
        if len(self.state) != self.ndim:
            raise ValueError("initial must be of dimension self.ndim = " +
                             str(self.ndim))
        self.out_mask = out_mask
        self.get_info = get_info
        self.log_every = log_every
        self.init_adapt(self.state)

    def next_state(self, state, iteration):
        """ Get the next state in the Markov chain.

        Depends on self.state, but must not change it.

        :return: The next state.
        """
        raise NotImplementedError("AbstractMarkovUpdate is abstract.")

    def sample(self, sample_size):
        """ Generate a sample of given size.

        :param sample_size: Number of samples to generate.
        :return: Numpy array with shape (sample_size, self.ndim).
        """
        # check if sampler was initialized
        if self.state is None:
            raise RuntimeError("Call init_sampler before sampling.")

        # only used if self.get_info is true.
        self.sample_info = SampleInfo()
        if self.get_info:
            self.sample_info.ndim = self.ndim
            self.sample_info.size = sample_size

        tagged = []
        tags = []
        tag_parser = None

        chain = np.empty((sample_size, self.ndim))
        chain[0] = self.state

        for i in range(1, sample_size):
            self.state = self.next_state(self.state, i)
            if self.get_info and not np.array_equal(self.state, chain[i - 1]):
                self.sample_info.accepted += 1

            chain[i] = self.state
            try:
                tags.append(self.state.tag)
                tagged.append(i)
                tag_parser = self.state.tag_parser
            except AttributeError:
                pass

            if self.log_every > 0 and (i + 1) % self.log_every == 0:
                print("Generated %d samples." % (i + 1))

        if self.out_mask is not None:
            chain = chain[:, self.out_mask]

        if self.get_info:
            self.sample_info.mean = np.mean(chain, axis=0)
            self.sample_info.var = np.var(chain, axis=0)

        if tagged:
            chain[tagged] = tag_parser(chain[tagged], tags)

        return chain


# METROPOLIS (HASTING) UPDATES
class MetropolisUpdate(AbstractMarkovUpdate):

    def __init__(self, ndim, target_pdf, is_adaptive=False):
        """ Generic abstract class to represent a single Metropolis-like update.
        """
        super().__init__(ndim, is_adaptive)
        self.pdf = target_pdf

        try:
            # if proposal_pdf is implemented, method is Metropolis Hasting
            self.proposal_pdf(np.zeros(ndim), np.zeros(ndim))
            self.is_hasting = True
        except (NotImplementedError, AttributeError):
            self.is_hasting = False

    def adapt(self, iteration, prev, state, accept):
        pass

    def accept(self, state, candidate):
        """ Default accept implementation for Metropolis/Hasting update.

        :param state: Previous state in the Markov chain.
        :param candidate: Candidate for next state.
        :return: The acceptance probability of a candidate state given the
            previous state in the Markov chain.
        """
        if self.is_hasting:
            return (candidate.pdf * self.proposal_pdf(candidate, state) /
                    state.pdf * self.proposal_pdf(state, candidate))
        else:
            # otherwise (simpler) Metropolis update
            return candidate.pdf / state.pdf

    def proposal(self, state):
        """ A proposal generator.

        Generate candidate points in the sample space.
        These are used in the update mechanism and
        accepted with a probability self.accept(candidate) that depends
        on the used algorithm.

        :param state: The previous state in the Markov chain.
        :return: A candidate state of type MetropolisState with pdf set.
        """
        raise NotImplementedError("MetropolisLikeUpdate is abstract.")

    def proposal_pdf(self, state, candidate):
        raise NotImplementedError("Implement for Hasting update.")

    def init_state(self, state):
        if not isinstance(state, MetropolisState):
            state = MetropolisState(state)
        if state.pdf is None:
            state.pdf = self.pdf(state)

        return super().init_state(state)

    def next_state(self, state, iteration):
        candidate = self.proposal(state)

        try:
            accept = self.accept(state, candidate)
        except (TypeError, AttributeError):
            # in situations like mixing/composite updates, previous update
            # may not have set necessary attributes (such as pdf)
            state = self.init_state(state)
            accept = self.accept(state, candidate)

        if accept >= 1 or np.random.rand() < accept:
            next_state = candidate
        else:
            next_state = state

        if self.is_adaptive:
            self.adapt(iteration, state, next_state, accept)

        return next_state


class CompositeMarkovUpdate(AbstractMarkovUpdate):

    def __init__(self, ndim, updates, masks=None):
        """ Composite Metropolis sampler; combine updates.

        :param updates: List of update mechanisms, each subtypes of
            MetropolisLikeUpdate.
        :param masks: Dictionary, giving masks (list/array of indices)
            of dimensions for the index of the update mechanism. Use this if
            some updates only affect slices of the state.
        """
        is_adaptive = any(update.is_adaptive for update in updates)
        super().__init__(ndim, is_adaptive)

        self.updates = updates
        self.masks = [None if masks is None or i not in masks else masks[i]
                      for i in range(len(updates))]

    def init_adapt(self, initial_state):
        for update in self.updates:
            update.init_adapt(initial_state)

    def next_state(self, state, iteration):
        for mechanism, mask in zip(self.updates, self.masks):
            if mask is None:
                state = mechanism.next_state(state, iteration)
            else:
                state = np.copy(state)
                state[mask] = mechanism.next_state(state[mask], iteration)

        return state


class MixingMarkovUpdate(AbstractMarkovUpdate):

    def __init__(self, ndim, updates, weights=None, masks=None):
        """ Mix a number of update mechanisms, choosing one in each step.

        :param updates: List of update mechanisms (AbstractMarkovUpdate).
        :param weights: List of weights for each of the mechanisms (sum to 1).
        :param masks: Slice object, specify if updates only affect slice of
            state.
        """
        is_adaptive = any(update.is_adaptive for update in updates)
        super().__init__(ndim, is_adaptive)

        self.updates = updates
        self.updates_count = len(updates)
        self.masks = [None if masks is None or i not in masks else masks[i]
                      for i in range(len(updates))]
        if weights is None:
            weights = np.ones(self.updates_count) / self.updates_count
        self.weights = weights

    def init_adapt(self, initial_state):
        for update in self.updates:
            update.init_adapt(initial_state)

    def next_state(self, state, iteration):
        index = np.random.choice(self.updates_count, p=self.weights)
        if self.masks[index] is None:
            return self.updates[index].next_state(state, iteration)
        else:
            mask = self.masks[index]
            state = np.copy(state)
            state[mask] = self.updates[index].next_state(state[mask], iteration)
            return state


def make_metropolis(ndim, target_pdf, proposal=None, proposal_pdf=None):
    """ Use the Metropolis algorithm to generate a sample.

    The dimensionality of the sample points is inferred from the length
    of initial.

    The proposal must not depend on the current state, if proposal_pdf
    is None (in that case simple Metropolis is used). If proposal is
    None, use Metropolis with a uniform proposal in [0,1].

    Example:
        >>> pdf = lambda x: np.sin(10*x)**2
        >>> met = make_metropolis(1, pdf)    # 1 dimensional
        >>> met.init_sampler(0.1)            # initialize with start value
        >>> sample = met.sample(1000)        # generate 1000 samples

    :param ndim: Dimensionality of sample space.
    :param target_pdf: Desired (unnormalized) probability distribution.
    :param proposal: A proposal generator. Take one or zero arguments.
        First argument is previous state. If it is used, and proposal is not
        symmetric, proposal_pdf must be passed.
    :param proposal_pdf: Function taking two arguments, previous and candidate
        state, and returns the conditional probability of proposing that
        candidate. Pass None (default) if the proposal is symmetric.
    """
    update = MetropolisUpdate(ndim, target_pdf)
    if proposal is None:
        def _proposal(_):
            """ Uniform proposal generator. """
            candidate = np.random.rand(ndim)
            return MetropolisState(candidate, target_pdf(candidate))

        if proposal_pdf is not None:
            raise RuntimeWarning("No proposal given, ignoring proposal_pdf")
        prop_pdf = None
    else:
        try:
            proposal()

            def prop(_):
                return proposal()

        except TypeError:
            prop = proposal

        def _proposal(state):
            """ Modified proposal """
            candidate = prop(state)
            return MetropolisState(candidate, target_pdf(candidate))

        try:
            proposal_pdf(np.zeros(ndim))

            def prop_pdf(_, candidate):
                return proposal_pdf(candidate)
        except TypeError:
            prop_pdf = proposal_pdf

    update.proposal = _proposal
    update.proposal_pdf = prop_pdf
    return update
