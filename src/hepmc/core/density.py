import numpy as np
from copy import copy

from .util import interpret_array


class _AnyDensity(object):
    # make Proposal and Density have the same __init__
    def __init__(self, ndim, is_symmetric=False):
        self.ndim = ndim
        self.is_symmetric = is_symmetric


class Proposal(_AnyDensity):

    def proposal(self, state):
        raise NotImplementedError()

    def proposal_pdf(self, state, candidate):
        return None  # may return None if is_symmetric == True

    @classmethod
    def make(cls, proposal, ndim=None, proposal_pdf=None, symmetric=None):
        if isinstance(proposal, Proposal):
            return proposal
        if ndim is None:
            raise RuntimeError("If proposal is a function, ndim must be given.")
        if symmetric is None:
            symmetric = proposal_pdf is None
        obj = cls(ndim, symmetric)
        obj.proposal = proposal
        if proposal_pdf is not None:
            obj.proposal_pdf = proposal_pdf
        return obj


class Density(_AnyDensity):

    def __call__(self, *xs):
        if np.isscalar(xs[0]):
            # xs are numbers
            return self.pdf(np.stack(xs, axis=0))
        else:
            shape = np.asanyarray(xs[0]).shape
            xs = [np.asanyarray(x).flatten() for x in xs]
            return self.pdf(np.stack(xs, axis=1)).reshape(shape)

    def pot(self, xs):
        with np.errstate(divide='ignore'):
            pot = -np.log(self.pdf(xs))

        pot[np.isnan(pot)] = np.inf
        return pot

    def pot_gradient(self, xs):
        xs = interpret_array(xs, self.ndim)
        pdf = self.pdf(xs)[:, np.newaxis]
        res = np.empty(xs.shape)

        # compute where pdf is != 0
        nonzero = (pdf != 0).flatten()
        pdf_grad = self.pdf_gradient(xs[nonzero])
        res[nonzero] = -pdf_grad / pdf[nonzero]

        # set others to inf
        res[np.logical_not(nonzero)] = np.inf

        return res

    def pdf(self, xs):
        raise NotImplementedError()

    def pdf_gradient(self, xs):
        raise NotImplementedError()

    @classmethod
    def make(cls, pdf=None, ndim=None, pdf_vect=None, is_symmetric=False,
             pdf_gradient=None, variance=None, mean=None):
        if isinstance(pdf, Density):
            obj = pdf
        else:
            if ndim is None:
                raise RuntimeError("If first argument is not a Density, "
                                   "ndim must be specified.")
            obj = cls(ndim, is_symmetric)

            if pdf is not None:
                obj.pdf = lambda xs: np.atleast_1d(pdf(xs)).flatten()
            elif pdf_vect is not None:
                obj.pdf = lambda xs: pdf_vect(*xs.transpose()).flatten()

        if pdf_gradient is not None:
            obj.pdf_gradient = pdf_gradient
        if variance is not None:
            obj.variance = variance
        if mean is not None:
            obj.mean = mean
        return obj


class Distribution(Density, Proposal):

    def pdf(self, xs):
        raise NotImplementedError()

    def pdf_gradient(self, xs):
        raise NotImplementedError()

    def rvs(self, sample_count):
        raise NotImplementedError()

    def proposal(self, state=None):
        return self.rvs(1)[0]

    def proposal_pdf(self, state, candidate):
        return float(self.pdf(candidate))

    @classmethod
    def make(cls, pdf=None, ndim=None, rvs=None, **kwargs):
        obj = super().make(pdf=pdf, ndim=ndim, **kwargs)
        if rvs is None:
            raise RuntimeWarning("Setting rvs with None.")

        obj.rvs = lambda count: interpret_array(rvs(count), ndim)
        return obj
