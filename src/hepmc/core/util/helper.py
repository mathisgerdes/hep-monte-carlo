import numpy as np


def assure_2d(array):
    """ Assure the vector is two dimensional.

    Several of the sampling algorithms work with samples of shape (N, ndim),
    if ndim is one it is, however, convenient to simply have the shape (N,).
    This method adds an extra dimension if necessary to allow both shapes.

    If the array has more than 2 dimensions, an exception is raised.

    Example:
        >>> x = np.array([0, 1, 2, 3])
        >>> y = assure_2d(x)
        >>> y.shape
        (4, 1)
        >>> y is assure_2d(y)  # y is already two dimensional
        True

    :param array: Numpy array, either one or two dimensional.
    :return: If the original shape is (N,) the returned array has shape (N, 1).
        For arrays with shape (N, ndim) this function is the identity.
    """
    array = np.array(array, copy=False, subok=True, ndmin=1)
    if array.ndim == 2:
        return array
    elif array.ndim == 1:
        return array[:, np.newaxis]
    else:
        raise RuntimeError("Array must be 1 or 2 dimensional.")


def interpret_array(array, ndim=None):
    array = np.array(array, copy=False, subok=True, ndmin=1)
    if array.ndim == 2:
        if ndim and ndim != array.shape[1]:
            raise RuntimeWarning("Unexpected dimension of entries in array.")
        return array

    if ndim is None:
        # can't distinguish 1D vs ND cases. Treat as 1D case
        return array[:, np.newaxis]

    if array.ndim == 1:
        if ndim == 1:
            # many 1D entries
            return array[:, np.newaxis]
        elif array.size == ndim:
            # one ndim-dim entry
            return array[np.newaxis, :]
        else:
            raise RuntimeError("Bad array shape.")


def damped_update(old, new, damping_onset, inertia):
    """ Update an old value given a new one, damping the change.

    The parameter inertia can be thought of loosely as the index of the change
    in a number of update iterations, where damping_onset specifies the
    damping-behavior. Both damping_onset and inertia can be floats.

    :param old:
    :param new:
    :param damping_onset: The value of inertia for which old and new values have
        equal weights in the update. Greater values mean smaller damping.
    :param inertia: Value > 0 specifying the strength of damping.
        In the limit to infinity, the old value is retained.
    :return: An updated value combining the old (current) an new values.
    """
    return (new * damping_onset / (inertia + 1 + damping_onset) +  # old
            old * (inertia + 1) / (inertia + 1 + damping_onset))   # new


def hypercube_bounded(index, null_value=0, shape=lambda xs: xs.shape[0],
                      self_has_ndim=False):
    """ Use as decorator if a function only returns none-null results [0,1]^dim.

    Includes performing interpret_array on xs.

    :param index: Index of the relevant value array xs.
    :param null_value: Value to set array to where xs==0.
    :param shape: Expression to get the return array shape given xs.
    :param self_has_ndim: True if the first funciton argument is self and
        self has the attribute ndim.
    :return: Numpy array with appropriate null-values and function values.
    """
    def get_bounded(fn):
        def fn_bounded(*args, **kwargs):
            xs = args[index]
            xs = interpret_array(xs, args[0].ndim if self_has_ndim else None)

            in_bounds = np.all((0 < xs) * (xs < 1), axis=1)

            res = np.empty(shape(xs))
            res[in_bounds] = fn(*args[:index], xs[in_bounds],
                                *args[index+1:], **kwargs)
            res[np.logical_not(in_bounds)] = null_value

            return res

        return fn_bounded
    return get_bounded


class Counted(object):
    def __init__(self, fn):
        self.fn = fn
        self.count = 0

    def __call__(self, xs, *args, **kwargs):
        xs_arr = np.asanyarray(xs)
        if xs_arr.ndim >= 2:
            self.count += xs_arr.shape[0]
        else:
            self.count += 1
        return self.fn(xs, *args, **kwargs)


def count_calls(obj, *method_names):
    for name in method_names:
        setattr(obj, name, Counted(getattr(obj, name)))
