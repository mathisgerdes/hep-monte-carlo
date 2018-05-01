import numpy as np
from .base import MarkovUpdate


class MetropolisState(np.ndarray):
    def __new__(cls, input_array, pdf=None):
        obj = np.array(input_array, copy=False, subok=True, ndmin=1).view(cls)
        obj.pdf = pdf
        return obj

    def __array_finalize__(self, obj):
        if obj is None:
            return  # was called from the __new__ above
        self.pdf = getattr(obj, 'pdf', None)


# METROPOLIS (HASTING) UPDATE
class MetropolisUpdate(MarkovUpdate):

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
                    state.pdf / self.proposal_pdf(state, candidate))
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


class DefaultMetropolis(MetropolisUpdate):

    def __init__(self, ndim, target_pdf, proposal=None, proposal_pdf=None):
        """ Use the Metropolis algorithm to generate a sample.

        Example:
            >>> pdf = lambda x: np.sin(10*x)**2
            >>> met = DefaultMetropolis(1, pdf)    # 1 dimensional
            >>> sample = met.sample(1000, 0.1)   # generate 1000 samples

        :param ndim: Dimensionality of sample space. :param target_pdf:
        Desired (unnormalized) probability distribution. :param proposal: A
        proposal generator. Take one or zero arguments. First argument is
        previous state. If it is used, and proposal is not symmetric,
        proposal_pdf must be passed. :param proposal_pdf: Function taking two
        arguments, previous and candidate state, and returns the conditional
        probability of proposing that candidate. Pass None (default) if the
        proposal is symmetric.
        """
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

                def _proposal(_):
                    return proposal()

            except TypeError:
                _proposal = proposal

            try:
                proposal_pdf(np.zeros(ndim))

                def prop_pdf(_, candidate):
                    return proposal_pdf(candidate)
            except TypeError:
                prop_pdf = proposal_pdf

        self._proposal = _proposal
        self._proposal_pdf = prop_pdf

        # must be at the and since it calls proposal_pdf to see if it works
        super().__init__(ndim, target_pdf, False)

    def proposal(self, state):
        candidate = self._proposal(state)
        return MetropolisState(candidate, self.pdf(candidate))

    def proposal_pdf(self, state, candidate):
        try:
            return self._proposal_pdf(state, candidate)
        except TypeError:
            raise NotImplementedError()
