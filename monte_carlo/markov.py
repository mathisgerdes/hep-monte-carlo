import numpy as np


# MARKOV CHAIN
class AbstractStepUpdate(object):
    """ Basic building block of a Markov chain. """

    def next_state(self, state):
        raise NotImplementedError("AbstractStepUpdate is abstract.")


# METROPOLIS (HASTING) UPDATES
class AbstractMetropolisUpdate(AbstractStepUpdate):
    """ Generic abstract class to represent a single Metropolis-like update.

    Does not hold information about a Markov chain but only about
    the update process used in the chain.
    """

    def accept(self, state, candidate):
        """ This function must be implemented by child classes.

        :param state: Previous state in the Markov chain.
        :param candidate: Candidate for next state.
        :return: The acceptance probability of a candidate state given the
            previous state in the Markov chain.
        """
        raise NotImplementedError("AbstractMetropolisUpdate is abstract.")

    def proposal(self, state):
        """ A proposal generator.

        Generate candidate points in the sample space.
        These are used in the update mechanism and
        accepted with a probability self.accept(candidate) that depends
        on the used algorithm.

        :param state: The previous state in the Markov chain.
        :return: A candidate state.
        """
        raise NotImplementedError("AbstractMetropolisUpdate is abstract.")

    def next_state(self, state):
        candidate = self.proposal(state)
        accept_prob = self.accept(state, candidate)
        if accept_prob >= 1 or np.random.rand() < accept_prob:
            return candidate
        return state


class MetropolisUpdate(AbstractMetropolisUpdate):

    def __init__(self, pdf, proposal, proposal_pdf=None):
        """ Metropolis (Hasting) update.

        :param pdf: Desired (unnormalized) probability distribution.
        :param proposal: A proposal generator.
            Takes the previous state as argument, but for this algorithm
            to work must not depend on it (argument exists only for generality
            of the implementation.)
        :param proposal_pdf: Conditional distribution of the candidate, given
            the state (state as first argument). If None, use the (simpler)
            Metropolis algorithm, otherwise Metropolis Hasting.
        """
        self._proposal = proposal
        self.is_hasting = proposal_pdf is not None
        self._proposal_pdf = proposal_pdf
        self.pdf = pdf
        self.proposal = proposal

    def accept(self, state, candidate):
        """ Probability of accepting candidate as next state. """
        if self.is_hasting:
            return (self.pdf(candidate) * self._proposal_pdf(candidate, state) /
                    self.pdf(state) / self._proposal_pdf(state, candidate))

        # otherwise Metropolis update
        return self.pdf(candidate) / self.pdf(state)

    def proposal(self, state):
        """ Propose a candidate state. """
        return self._proposal(state)


# CONCRETE SAMPLERS (contain state of the markov chain, allow sampling)
class AbstractMetropolisSampler(object):

    def __init__(self, initial):
        """ Generic Metropolis (Hasting) sampler.

        The dimensionality of the sample points is inferred from the length
        of initial.

        Class is abstract, child class has to implement a function 'accept'.
        Function, takes the previous and next state and returns
        the acceptance probability. (Values greater than 1 are treated as 1.)

        :param initial: Initial value of the Markov chain. Numpy array.
        """
        initial = np.array(initial, copy=False, subok=True, ndmin=1)
        self.state = initial
        self.dim = len(initial)

    def next_state(self, state):
        """ Get the next state in the Markov chain.

        Depends on self.state, but must not change it.

        :return: The next state
        """
        raise NotImplementedError("AbstractMetropolisSampler is abstract.")

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
            next_state = self.next_state(self.state)
            if get_accept_rate and not np.array_equal(next_state, self.state):
                accepted += 1
            chain[i] = self.state = next_state

        if get_accept_rate:
            return chain, accepted / sample_size
        return chain


class MetropolisSampler(MetropolisUpdate, AbstractMetropolisSampler):

    def __init__(self, initial, pdf, proposal=None, proposal_pdf=None):
        """ Use the Metropolis algorithm to generate a sample.

        The dimensionality of the sample points is inferred from the length
        of initial.

        The proposal must not depend on the current state, if proposal_pdf
        is None (in that case simple Metropolis is used). If proposal is
        None, use Metropolis with a uniform proposal in [0,1].

        Example:
            >>> pdf = lambda x: np.sin(10*x)**2
            >>> met = MetropolisSampler(0.1, pdf)
            >>> sample = met(1000)

        :param initial: Initial value of the Markov chain. Internally
            converted to numpy array.
        :param pdf: Desired (unnormalized) probability distribution.
        :param proposal: A proposal generator.
            Takes the previous state as argument, but for this algorithm
            to work must not depend on it (argument exists only for generality
            of the implementation.)
        """
        if proposal is None:
            def proposal(_):
                """ Uniform proposal generator. """
                return np.random.rand(self.dim)

        MetropolisUpdate.__init__(self, pdf, proposal, proposal_pdf)
        AbstractMetropolisSampler.__init__(self, initial)


class CompositeMetropolisSampler(AbstractMetropolisSampler):

    def __init__(self, initial, mechanisms):
        """ Composite Metropolis sampler; combine updates.

        :param mechanisms: List of update mechanisms, each subtypes of
            AbstractMetropolisUpdate
        """
        AbstractMetropolisSampler.__init__(self, initial)
        self.mechanisms = mechanisms

    def next_state(self, state):
        for mechanism in self.mechanisms:
            state = mechanism.next_state(state)

        return state


class MixingMetropolisSampler(AbstractMetropolisSampler):

    def __init__(self, initial, mechanisms, weights=None):
        """ Mix a number of update mechanisms, choosing one in each step.

        :param initial: Initial value.
        :param mechanisms: List of update mechanisms (AbstractStepUpdate).
        :param weights: List of weights for each of the mechanisms (sum to 1).
        """
        AbstractMetropolisSampler.__init__(self, initial)
        self.mechanisms = mechanisms
        self.weights = weights

    def next_state(self, state):
        mechanism = np.random.choice(self.mechanisms, 1, p=self.weights)
        return mechanism.next_state(state)
