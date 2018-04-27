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
