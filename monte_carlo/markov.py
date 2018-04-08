import numpy as np


# MARKOV CHAIN
class AbstractStepUpdate(object):
    """ Basic building block of a Markov chain. """

    def next_state(self, state):
        raise NotImplementedError("AbstractStepUpdate is abstract.")


# METROPOLIS (HASTING) UPDATES
class AbstractMetropolisUpdate(AbstractStepUpdate):
    """ Generic abstract class to represent a single Metropolis update.

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


class MetropolisHastingUpdate(AbstractMetropolisUpdate):

    def __init__(self, pdf, proposal_pdf, proposal):
        """ Metropolis Hasting update.

        :param pdf: Desired (unnormalized) probability distribution.
        :param proposal: A proposal generator.
            Takes the previous state as argument, but for this algorithm
            to work must not depend on it (argument exists only for generality
            of the implementation.)
        """
        self._proposal = proposal
        self._proposal_pdf = proposal_pdf
        self.pdf = pdf
        self.proposal = proposal

    def accept(self, state, candidate):
        """ Probability of accepting candidate as next state. """
        return (self.pdf(candidate) * self._proposal_pdf(candidate, state) /
                self.pdf(state) / self._proposal_pdf(state, candidate))

    def proposal(self, state):
        """ Propose a candidate state. """
        return self._proposal(state)


class MetropolisUpdate(MetropolisHastingUpdate):

    def __init__(self, pdf, proposal):
        """ Metropolis update.

        :param pdf:  Desired (unnormalized) probability distribution.
        :param proposal: A proposal generator.
            Takes the previous state as argument, but for this algorithm
            to work must not depend on it (argument exists only for generality
            of the implementation.)
        """
        # proposal_pdf is not required because proposal is symmetric
        MetropolisHastingUpdate.__init__(self, pdf, None, proposal)

    def accept(self, state, candidate):
        """ Probability of accepting candidate as next state. """
        return self.pdf(candidate) / self.pdf(state)


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

    def __init__(self, initial, pdf, proposal=None):
        """ Use the Metropolis algorithm to generate a sample.

        The dimensionality of the sample points is inferred from the length
        of initial.

        The proposal must not depend on the current state.
        Use the Metropolis Hasting algorithm if it does.

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

        initial = np.array(initial, copy=False, subok=True, ndmin=1)
        MetropolisUpdate.__init__(self, pdf, proposal)
        AbstractMetropolisSampler.__init__(self, initial)


class MetropolisHastingSampler(MetropolisHastingUpdate,
                               AbstractMetropolisSampler):

    def __init__(self, initial, pdf, proposal_pdf=None, proposal=None):
        """ Metropolis Hasting sampler.

        Dimensionality is inferred from the length of initial.

        If proposal_pdf and proposal are not specified (None),
        a uniform distribution is used. This makes the method equivalent to
        the simpler Metropolis algorithm as the candidate distribution does
        not depend on the previous state.

        proposal_pdf and proposal must either both be specified or both None.

        :param initial: Initial value of the Markov chain (array-like).
        :param pdf: Desired (unnormalized) probability distribution.
        :param proposal: Function that generates a candidate state, given
            the previous state as single argument.
        :param proposal_pdf: Distribution of a proposed candidate.
            Takes the previous state and the generated candidate as arguments.
        """
        initial = np.array(initial, copy=False, subok=True, ndmin=1)

        if proposal_pdf is None and proposal is None:
            dim = len(initial)

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

        MetropolisHastingUpdate.__init__(self, pdf, proposal_pdf, proposal)
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
