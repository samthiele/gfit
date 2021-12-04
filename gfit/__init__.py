import numpy as np
from scipy.optimize import least_squares
import numba
from numba import jit
from numba import prange
import math

### numpy implementation of multigauss function [ slower than numba + pure python ]
# @jit(nopython=True)
# def amgauss(x, y, a, b, c1, c2):
#     """
#     Evaluate and sum a set of asymmetric gaussian functions.
#
#     *Arguments*:
#      - x = array of x-values to evaluate gaussians over.
#      - y = array to put output y-values in (avoids doing memory allocation).
#      - a = a list of kernel sizes.
#      - b = a list of kernel centers.
#      - c1 = a list of left-hand kernel widths.
#      - c2 = a list of right-hand kernel widths.
#     """
#
#     y *= 0  # avoid doing memory allocation here
#     for _a, _b, _c1, _c2 in zip(a, b, c1, c2):
#         idx = np.argmax(x > _b)  # split point
#         y[:idx] += _a * np.exp(-(x[:idx] - _b) ** 2 / _c1)
#         y[idx:] += _a * np.exp(-(x[idx:] - _b) ** 2 / _c2)
#
# @jit(nopython=True)
# def mgauss(x, y, a, b, c):
#     """
#     Evaluate and sum a set of symmetric gaussian functions.
#
#     *Arguments*:
#      - x = array of x-values to evaluate gaussians over.
#      - y = array to put output y-values in (avoids doing memory allocation).
#      - a = a list of kernel sizes.
#      - b = a list of kernel centers.
#      - c = a list of kernel widths.
#     """
#     y *= 0  # avoid doing memory allocation here
#     for _a, _b, _c in zip(a, b, c):
#         y += _a * np.exp(-(x - _b) ** 2 / _c)

#################################
## Multi-gaussian model
#################################

### implementation no numpy
@jit(nopython=True)
def amgauss(x, y, a, b, c1, c2):
    """
    Evaluate and sum a set of asymmetric gaussian functions.

    *Arguments*:
     - x = array of x-values to evaluate gaussians over.
     - y = array to put output y-values in (avoids doing memory allocation).
     - a = a list of kernel sizes.
     - b = a list of kernel centers.
     - c1 = a list of left-hand kernel widths.
     - c2 = a list of right-hand kernel widths.
    """
    for i in range(len(x)):  # loop through points in vector
        y[i] = 0  # zero
        for j in range(len(a)):  # loop through gaussians and accumulate
            if c1[j] == 0 or c2[j] == 0:  # skip zeros
                continue
            if x[i] < b[j]:  # which width do we use?
                y[i] += a[j] * math.exp(-(x[i] - b[j]) ** 2 / c1[j])
            else:
                y[i] += a[j] * math.exp(-(x[i] - b[j]) ** 2 / c2[j])


@jit(nopython=True)
def mgauss(x, y, a, b, c):
    """
    Evaluate and sum a set of symmetric gaussian functions.

    *Arguments*:
     - x = array of x-values to evaluate gaussians over.
     - y = array to put output y-values in (avoids doing memory allocation).
     - a = a list of kernel sizes.
     - b = a list of kernel centers.
     - c = a list of kernel widths.
    """
    for i in range(len(x)):  # loop through points in vector
        y[i] = 0  # zero
        for j in range(len(a)):  # loop through gaussians and accumulate
            if c[j] == 0:  # skip zeros
                continue
            y[i] += a[j] * math.exp(-(x[i] - b[j]) ** 2 / c[j])


#################################
## Associated jacobians
#################################

@jit(nopython=True)
def amgauss_J(x, J, a, b, c1, c2):
    """
    Evaluate return the Jacobian to amgauss.

    *Arguments*:
     - x = array of x-values to evaluate gaussians over.
     - J = (n,4) array to put output jacobian in (avoids doing memory allocation).
     - a = a list of kernel sizes.
     - b = a list of kernel centers.
     - c1 = a list of left-hand kernel widths.
     - c2 = a list of right-hand kernel widths.
    """
    for i in range(len(x)):  # loop through points in vector
        for j in range(len(a)):  # loop through gaussians and accumulate
            if c1[j] == 0 or c2[j] == 0:  # skip zeros as these cause infs
                continue
            if x[i] < b[j]:  # which width do we use?
                J[i, 4 * j] = -math.exp(-(x[i] - b[j]) ** 2 / c1[j])
                J[i, 4 * j + 1] = ((2 * a[j] * (x[i] - b[j])) / c1[j]) * J[i, 4 * j]
                J[i, 4 * j + 2] = ((a[j] * (x[i] - b[j]) ** 2) / c1[j] ** 2) * J[i, 4 * j]
                J[i, 4 * j + 3] = 0
            else:
                J[i, 4 * j] = -math.exp(-(x[i] - b[j]) ** 2 / c1[j])
                J[i, 4 * j + 1] = ((2 * a[j] * (x[i] - b[j])) / c1[j]) * J[i, 4 * j]
                J[i, 4 * j + 2] = 0
                J[i, 4 * j + 3] = ((a[j] * (x[i] - b[j]) ** 2) / c1[j] ** 2) * J[i, 4 * j]


@jit(nopython=True)
def mgauss_J(x, J, a, b, c):
    """
    Evaluate and return the jacobian to mgauss.

    *Arguments*:
     - x = array of x-values to evaluate gaussians over.
     - J = (n,3) array to put output jacobian in (avoids doing memory allocation). This is where the output will be stored.
     - a = a list of kernel sizes.
     - b = a list of kernel centers.
     - c = a list of kernel widths.
    """
    for i in range(len(x)):  # loop through points in vector
        for j in range(len(a)):
            if c[j] == 0:  # skip zeros as these cause infs
                continue
            # evaluate partial derivatives
            J[i, j * 3] = -math.exp(-(x[i] - b[j]) ** 2 / c[j])  # -e^(-(x-b)^2/c)
            J[i, j * 3 + 1] = (2 * a[j] * (x[i] - b[j]) * J[i, j * 3]) / c[j]  # ((2 a (-b + x)) e^(-(-b + x)^2/c) ) / c
            J[i, j * 3 + 2] = (a[j] * ((x[i] - b[j]) ** 2) * J[i, j * 3]) / c[j] ** 2  # (a (-b + x)^2 e^(-(x-b)^2/c) ) / c^2

#################################
## Initialization routines
#################################
@jit(nopython=True)
def est_peaks(x, y, n, sym=True, d=10):
    """
    Find the n-largest peaks and use these to create an initial guess
    for the multigauss function.

    *Arguments*:
     - x = the x-coordinates of the y-values (used for calculating positions only).
     - y = the y-values to fit.
     - n = the number of features/peaks to detect.
     - sym = return asymmetric features. Default is True.
     - d = the distance to check for peak shoulders. Default is 5.
    *Returns*:
     - a = a list of n feature heights.
     - b = a list of n feature positions.
     - c1 = a list of n feature widths.
     - c2 = a list of n feature widths. If sym=True then this will be identical to c1.
    """

    # declare variables
    a = np.zeros(n)
    b = np.zeros(n)
    c1 = np.zeros(n)
    c2 = np.zeros(n)

    # loop through y and find peaks
    for i in range(len(y)):

        # check if this is a local maxima
        mx = True
        for j in range(d + 1):
            if y[i] < y[max(i - j, 0)] or y[i] < y[min(i + j, len(y) - 1)]:
                mx = False  # nope - skip to next point

        # this is a local maxima
        if mx:
            for j in range(n):
                if y[i] > a[j]:  # this is a larger peak than previously found ones

                    # push previous values down a place
                    for k in np.arange(j + 1, n)[::-1]:
                        a[k] = a[k - 1]
                        b[k] = b[k - 1]
                        c1[k] = c1[k - 1]
                        c2[k] = c2[k - 1]

                    a[j] = y[i]  # store height
                    b[j] = x[i]  # store position
                    # N.B. For normal distribution, 90th % Z = 1.64 sigma, hence normalization factor
                    c1[j] = y[i] * (
                        abs(x[i] - x[max(i - d, 0)] / y[i] - y[max(i - d, 0)])) / 1.64  # left-hand intercept
                    c2[j] = y[i] * (abs(x[i] - x[min(i + d, len(y) - 1)] / y[i] - y[
                        min(i + d, len(y) - 1)])) / 1.64  # right-hand intercept

                    break  # all done

    if sym:  # return symmetric
        c1 = (c1 + c2) / 2
        return a, b, c1, c1
    else:  # return assymetric
        return a, b, c1, c2


@jit(nopython=True, parallel=True)
def initialise(x, X, n, sym=True, d=10, nthreads=-1):
    """
    Compute initial estimates of gaussian positions, widths and heights for a vector of spectra / signals.

    *Arguments*:
     - x = a (n,) array containing the x-values of the spectra / signals.
     - X = a (m,n) array containing corresponding y-values for m different spectra / signals.
     - n = the number of gaussians to fit.
     - sym = True if symmetric gaussians should be fit. If this is the case, the two returned widths will be identical.
    """

    # init output array
    if sym:
        out = np.zeros((X.shape[0], n * 3))  # scale, position, width for each feature
    else:
        out = np.zeros((X.shape[0], n * 4))  # scale, position, width_L, width_R for each feature

    # setup multithreading
    if n != -1:  # -1 uses numba default
        t = numba.get_num_threads()  # store so we set this back later
        numba.set_num_threads(n)

    # loop through spectra
    for i in prange(X.shape[0]):

        # find initial values
        x0 = est_peaks(x, X[i, :], sym=sym, n=n, d=d)

        # store them
        for j in range(n):
            if sym:
                out[i, j * 3] = x0[0][j]
                out[i, j * 3 + 1] = x0[1][j]
                out[i, j * 3 + 2] = x0[2][j]
            else:
                out[i, j * 4] = x0[0][j]
                out[i, j * 4 + 1] = x0[1][j]
                out[i, j * 4 + 2] = x0[2][j]
                out[i, j * 4 + 3] = x0[3][j]

    # reset default nthreads
    if n != -1:
        numba.set_num_threads(t)

    # return
    return out


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


#################################
## Least squares fitting
#################################
@jit(nopython=True)
def lsq_amg(params, x, y, m, J ):
    """
    Calculate residual for least squares optimization of an asymmetric multigauss function.
    """
    # evaluate function
    amgauss(x, m, params[::4],params[1::4],params[2::4],params[3::4])
    return y - m # return residual

@jit(nopython=True)
def lsq_mg(params, x, y, m, J ):
    """
    Calculate residual for least squares optimization of a symmetric multigauss function.
    """
    mgauss(x, m, params[::3],params[1::3],params[2::3])
    return y - m

@jit(nopython=True)
def lsq_Jamg(params, x, y, m, J ):
    """
    Calculate and return jacobian for least-squares optimization of asymmetric multigauss function.
    """
    amgauss_J(x, J, params[::4],params[1::4],params[2::4],params[3::4])
    return J

@jit(nopython=True)
def lsq_Jmg(params, x, y, m, J ):
    """
    Calculate and return jacobian for least-squares optimization of symmetric multigauss function.
    """
    mgauss_J(x, J, params[::3],params[1::3],params[2::3])
    return J


def fit_mgauss(x, y, x0, n, c=(-np.inf, np.inf), ftol=1e-1, xtol=0, scale=0, maxiter=15, verbose=False):
    """
    Fit a symmetric multigauss to the provided 1-D data.

    *Arguments*:
     - x = x values of data to fit.
     - y = y values of data to fit.
     - x0 = the initial guess as a stacked (1d) array [see stack_coeff].
     - n = number of gaussians to fit.
     - c = a tuple of contstrains like ([x00min,x01min,...,x0nmin],[x00max,x01max,...,x0nmax]). Set as None to use no constraints.
     - ftol = ftol parameter of scipy.optimize.least_squares
    """

    # compute defaults if needed
    if scale == 0:
        scale = x[1] - x[0]
    if xtol == 0:
        xtol = scale / 100

    # refine inital guess using subset of data
    m = np.zeros_like(x)  # models will be evaluated here [ to save lots of memory allocations ]
    J = np.zeros((len(x), n * 3))
    fit = least_squares(lsq_mg, x0=x0, args=(x, y, m, J), jac=lsq_Jmg,
                        x_scale=scale, bounds=c, verbose=verbose,
                        ftol=ftol, xtol=xtol, max_nfev=maxiter)
    return fit.x


def fit_amgauss(x, y, x0, n, c=(-np.inf, np.inf), ftol=1e-1, xtol=0, scale=0, maxiter=15, verbose=False):
    """
    Fit a asymmetric multigauss to the provided 1-D data.

    *Arguments*:
     - x = x values of data to fit.
     - y = y values of data to fit.
     - x0 = the initial guess as a stacked (1d) array [see stack_coeff].
     - n = number of gaussians to fit.
     - c = a tuple of contstrains like ([x00min,x01min,...,x0nmin],[x00max,x01max,...,x0nmax]). Set as None to use no constraints.
     - ftol = ftol parameter of scipy.optimize.least_squares
    """

    # compute defaults if needed
    if scale == 0:
        scale = x[1] - x[0]
    if xtol == 0:
        xtol = scale / 100

    # refine inital guess using subset of data
    m = np.zeros_like(x)  # models will be evaluated here [ to save lots of memory allocations ]
    J = np.zeros((len(x), n * 4))
    fit = least_squares(lsq_amg, x0=x0, args=(x, y, m, J), jac=lsq_Jamg,
                        x_scale=scale, bounds=c, verbose=verbose,
                        ftol=ftol, xtol=xtol, max_nfev=maxiter)
    return fit.x