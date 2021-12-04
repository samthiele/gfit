"""
Utility functions for e.g. stacking / splitting parameter arrays or estimating bounds.
"""

import numpy as np

def stack_coeff(a, b, c1, c2=None):
    """
    Stack separate coefficient arrays into one matrix.

    *Arguments*:
     - a,b,c1,[c2] = each of the coefficients to stack. Must be numpy arrays. If c2 is None then
                   this will return a symetric coefficient matrix of (a,b,c) rather than (a,b,c2,c2).
    *Returns*:
     - a numpy array of coefficients.
    """
    if isinstance(a, list) or len(a.shape) == 0:  # 1-D case
        if c2 is not None:
            return np.array([a, b, c1, c2]).ravel(order='F')
        else:
            return np.array([a, b, c1]).ravel(order='F')
    else:  # 2-D case
        if c2 is not None:
            return np.vstack([a, b, c1, c2]).reshape(a.shape[:-1] + (a.shape[-1] * 4,), order='F')
        else:
            return np.vstack([a, b, c1]).reshape(a.shape[:-1] + (a.shape[-1] * 3,), order='F')


def split_coeff(arr, sym=False):
    """
    Split a numpy array containing a stack of coefficients into separate lists.

    *Arguments*:
     - arr = the numpy array to split.
     - sym = True if arr contains symmetric data (only one width) or assymetric (two widths; default).
    *Returns*:
     - a,b,c1,[c2] = the separated arrays of coefficients
    """
    if sym:
        assert (arr.shape[-1] % 3) == 0, "Error - array has an invalid shape %s" % str(arr.shape)
        n = int(arr.shape[-1] / 3)
        if len(arr.shape) == 1:  # 1D case; split individual signal
            a = np.array([arr[i * 3] for i in range(n)])
            b = np.array([arr[i * 3 + 1] for i in range(n)])
            c1 = np.array([arr[i * 3 + 2] for i in range(n)])
            return a.ravel(), b.ravel(), c1.ravel()
        else:  # N-D case; split last axis
            a = np.array([arr[..., i * 3] for i in range(n)]).T
            b = np.array([arr[..., i * 3 + 1] for i in range(n)]).T
            c1 = np.array([arr[..., i * 3 + 2] for i in range(n)]).T
            return a, b, c1
    else:
        assert (arr.shape[-1] % 4) == 0, "Error - array has an invalid shape %s" % str(arr.shape)
        n = int(arr.shape[-1] / 4)
        if len(arr.shape) == 1:  # 1D case; split individual signal
            a = np.array([arr[i * 4] for i in range(n)])
            b = np.array([arr[i * 4 + 1] for i in range(n)])
            c1 = np.array([arr[i * 4 + 2] for i in range(n)])
            c2 = np.array([arr[i * 4 + 3] for i in range(n)])
            return a.ravel(), b.ravel(), c1.ravel(), c2.ravel()
        else:  # N-D case; split last axis
            a = np.array([arr[..., i * 4] for i in range(n)]).T
            b = np.array([arr[..., i * 4 + 1] for i in range(n)]).T
            c1 = np.array([arr[..., i * 4 + 2] for i in range(n)]).T
            c2 = np.array([arr[..., i * 4 + 3] for i in range(n)]).T
            return a, b, c1, c2

def get_bounds(x, x0, sym=False, xpad=0.1, hpad=0.5):
    """
    Return lower and upper bounds to use for optimization based on initial guess.

    *Arguments*:
     - x = the x-domain to evaluate over.
     - x0 = the initial guess as returned by initialize( ... ).
     - sym = True if x0 contains features for symmetric gaussians (3 vars per feature). Default is False.
     - xpad = padding to min(x) and max(x) to allow features centered outside of range of x.
     - hpad = amount of height increase allowed relative to largest height (a) in x0.
    """

    # split
    if sym:
        assert (x0.shape[-1] % 3) == 0, "Error - array has an invalid shape %s" % str(x0.shape)
        fa, fb, _ = split_coeff(x0, sym=True)
        n = int(x0.shape[-1] / 3)
    else:
        assert (x0.shape[-1] % 4) == 0, "Error - array has an invalid shape %s" % str(x0.shape)
        fa, fb, _, _ = split_coeff(x0, sym=False)
        n = int(x0.shape[-1] / 4)

    mnx = np.min(x) - xpad
    mxx = np.max(x) + xpad
    mxh = np.max(fa) * (1 + hpad)

    if sym:
        return np.array([[0, mnx, 0] * n, [mxh, mxx, np.inf] * n])
    else:
        return np.array([[0, mnx, 0, 0] * n, [mxh, mxx, np.inf, np.inf] * n])
