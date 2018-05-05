import numpy as np

from ..util import interpret_array


class Density(object):

    def __init__(self, ndim, is_symmetric=None):
        self.ndim = ndim
        self.is_symmetric = is_symmetric

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

    @property
    def variance(self):
        raise AttributeError()

    @property
    def mean(self):
        raise AttributeError()


class Distribution(Density):

    def proposal(self):
        return self.rvs(1)[0]

    def pdf(self, xs):
        raise NotImplementedError()

    def pdf_gradient(self, xs):
        raise NotImplementedError()

    def rvs(self, sample_count):
        raise NotImplementedError()


def as_dist(pdf, ndim=None, rvs=None, pdf_gradient=None,
            variance=None, mean=None):
    if isinstance(pdf, Density):
        dist = pdf
    else:
        if ndim is None:
            raise RuntimeError("If first argument is not a Density, "
                               "ndim must be specified.")

        if rvs is None:
            dist = Density(ndim)
        else:
            dist = Distribution(ndim)
            dist.rvs = lambda count: interpret_array(rvs(count), ndim)

        dist.pdf = lambda xs: pdf(xs).flatten()

    if pdf_gradient is not None:
        dist.pdf_gradient = pdf_gradient
    if variance is not None:
        dist.variance = variance
    if mean is not None:
        dist.mean = mean
    return dist


def as_dist_vect(pdf_vect, **kwargs):
    def pdf(xs):
        return pdf_vect(*xs.transpose()).flatten()

    return as_dist(pdf, **kwargs)
