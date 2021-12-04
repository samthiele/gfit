"""
Functions used internally by gfit but not (really) intended for public scope.
"""

import numpy as np
from scipy.optimize import least_squares
from numba import jit
from .util import get_bounds
import math
from tqdm import tqdm

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

#########################################################
## Jacobians associated with multi-gauss functions
#########################################################

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
    b = np.full(n, np.nan)
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

    # replace zeros in position with x[ len(x) / 2 ]
    for i in range(len(b)):
        if np.isnan(b[i]):
            b[i] = x[ int(len(x) / 2) ]

    if sym:  # return symmetric
        c1 = (c1 + c2) / 2
        return a, b, c1, c1
    else:  # return assymetric
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

    # check bounds
    if not check_bounds( x0, c[0], c[1] ):
        print("\nInvalid bounds or initial guess:")
        print("   x0 = %s" % x0)
        print("   c = %s" % c)

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

    # check bounds
    if not check_bounds( x0, c[0,:], c[1,:] ):
        print("\nInvalid bounds or initial guess:")
        print("   x0 = %s" % x0)
        print("   c = %s" % c)

    # refine inital guess using subset of data
    m = np.zeros_like(x)  # models will be evaluated here [ to save lots of memory allocations ]
    J = np.zeros((len(x), n * 4))
    fit = least_squares(lsq_amg, x0=x0, args=(x, y, m, J), jac=lsq_Jamg,
                        x_scale=scale, bounds=c, verbose=verbose,
                        ftol=ftol, xtol=xtol, max_nfev=maxiter)
    return fit.x

@jit(nopython=True)
def check_bounds( x0, mn, mx):
    """
    Checks that bounds are valid, and asserts False with a usable warning message if not.
    """
    for i in range(len(x0)):
        if not (mn[i] <= x0[i] and mx[i] >= x0[i]):
            return False
    return True

def gfit_single(x, X, x0, n, sym=True, nthreads=-1, vb=True, **kwds ):
    """ Single-threaded multigaussian fitting"""

    # wrap X and x0 if needed
    if len(X.shape) == 1:
        X = np.array([X])
    if len(x0.shape) == 1:
        x0 = np.array([x0])

     # create output
    out = np.zeros_like(x0)

    # which function to use
    if sym:
        _opt = fit_mgauss
    else:
        _opt = fit_amgauss

    # get bounds
    c = kwds.pop("c", get_bounds(x, x0, sym=sym))

    # loop through all data points
    loop = range(0,X.shape[0])
    if vb:
        loop = tqdm(loop, desc='Fitting gaussians',leave=False)
    for i in loop:
        out[i,:] = _opt(x,X[i,:],x0[i,:],n,c)

    # return
    return out

def gfit_multi(x, X, x0, n, sym=True, nthreads=-1, vb=True, **kwds ):
    """ Multi-threaded multigaussian fitting"""
    pass # TODO

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