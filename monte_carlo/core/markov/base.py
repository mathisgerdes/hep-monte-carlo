import numpy as np
from ..sampling import Sample


class MarkovSample(Sample):

    def __init__(self, **kwargs):
        self.accepted = 0

        super().__init__(**kwargs)

        self._sample_info.append(('accept_ratio', 'acceptance rate', '%f'))

    @property
    def accept_ratio(self):
        return self.accepted / self.size


# MARKOV CHAIN
class MarkovUpdate(object):
    """ Basic update mechanism of a Markov chain. """

    def __init__(self, ndim, is_adaptive=False, target=None):
        self.ndim = ndim
        self.target = target
        self.is_adaptive = is_adaptive

        # will hold information if update was used as a sampler
        self.sample_info = None

    def init_adapt(self, initial_state):
        pass

    def init_state(self, state):
        return state  # may initialize other state attributes (such as pdf)

    def next_state(self, state, iteration):
        """ Get the next state in the Markov chain.

        :return: The next state.
        """
        raise NotImplementedError("AbstractMarkovUpdate is abstract.")

    def sample(self, sample_size, initial, out_mask=None, log_every=5000):
        """ Generate a sample of given size.

        :param sample_size: Number of samples to generate.
        :param initial: Initial value of the Markov chain. Internally
            converted to numpy array.
        :param out_mask: Slice object, return only this slice of the output
            chain (useful if sampler uses artificial variables).
        :param log_every: Print the number of generated samples. Do not log if
            value is < 0. Log every sample for log=1.
        :return: Numpy array with shape (sample_size, self.ndim).
        """
        # initialize sampling
        state = self.init_state(np.atleast_1d(initial))
        if len(state) != self.ndim:
            raise ValueError('initial must have dimension ' + str(self.ndim))
        self.init_adapt(state)  # initial adaptation

        sample = MarkovSample()

        tags = dict()
        tagged = dict()

        chain = np.empty((sample_size, self.ndim))
        chain[0] = state

        for i in range(1, sample_size):
            state = self.next_state(state, i)
            if not np.array_equal(state, chain[i - 1]):
                sample.accepted += 1

            chain[i] = state
            try:
                try:
                    tags[state.tag_parser].append(state.tag)
                    tagged[state.tag_parser].append(i)
                except KeyError:
                    tags[state.tag_parser] = []
                    tagged[state.tag_parser] = []
            except AttributeError:
                pass

            if log_every > 0 and (i + 1) % log_every == 0:
                print("Generated %d samples." % (i + 1), flush=True)

        if out_mask is not None:
            chain = chain[:, out_mask]

        for parser in tagged:
            chain[tagged[parser]] = parser(chain[tagged[parser]], tags[parser])

        sample.data = chain
        sample.target = self.target
        return sample


class CompositeMarkovUpdate(MarkovUpdate):

    def __init__(self, ndim, updates, masks=None, target=None):
        """ Composite Markov update; combine updates.

        :param updates: List of update mechanisms, each subtypes of
            MetropolisLikeUpdate.
        :param masks: Dictionary, giving masks (list/array of indices)
            of dimensions for the index of the update mechanism. Use this if
            some updates only affect slices of the state.
        """
        is_adaptive = any(update.is_adaptive for update in updates)
        if target is None:
            for update in updates:
                if update.target is not None:
                    target = update.target
                    break
        super().__init__(ndim, is_adaptive=is_adaptive, target=target)

        self.updates = updates
        self.masks = [None if masks is None or i not in masks else masks[i]
                      for i in range(len(updates))]

    def init_adapt(self, initial_state):
        for update in self.updates:
            state = update.init_state(initial_state)
            update.init_adapt(state)

    def next_state(self, state, iteration):
        for mechanism, mask in zip(self.updates, self.masks):
            if mask is None:
                state = mechanism.next_state(state, iteration)
            else:
                state = np.copy(state)
                state[mask] = mechanism.next_state(state[mask], iteration)

        return state


class MixingMarkovUpdate(MarkovUpdate):

    def __init__(self, ndim, updates, weights=None, masks=None, target=None):
        """ Mix a number of update mechanisms, choosing one in each step.

        :param updates: List of update mechanisms (AbstractMarkovUpdate).
        :param weights: List of weights for each of the mechanisms (sum to 1).
        :param masks: Slice object, specify if updates only affect slice of
            state.
        """
        is_adaptive = any(update.is_adaptive for update in updates)
        if target is None:
            for update in updates:
                if update.target is not None:
                    target = update.target
                    break
        super().__init__(ndim, is_adaptive=is_adaptive, target=target)

        self.updates = updates
        self.updates_count = len(updates)
        self.masks = [None if masks is None or i not in masks else masks[i]
                      for i in range(len(updates))]
        if weights is None:
            weights = np.ones(self.updates_count) / self.updates_count
        self.weights = weights

    def init_adapt(self, initial_state):
        for update in self.updates:
            state = update.init_state(initial_state)
            update.init_adapt(state)

    def next_state(self, state, iteration):
        index = np.random.choice(self.updates_count, p=self.weights)
        if self.masks[index] is None:
            state = self.updates[index].init_state(state)
            return self.updates[index].next_state(state, iteration)
        else:
            mask = self.masks[index]
            state = np.copy(state)
            state[mask] = self.updates[index].next_state(state[mask], iteration)
            return state
