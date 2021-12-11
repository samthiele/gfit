"""
Functions used internally by gfit but not (really) intended for public scope.
"""

import os

# disable numpy multi-threading; we handle this.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from scipy.optimize import least_squares
import numba
from numba import jit, prange
from .util import get_bounds
import math
from tqdm import tqdm


#################################
## Multi-gaussian model
#################################

### implementation no numpy
@jit(nopython=True,nogil=True,cache=True)
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


@jit(nopython=True,nogil=True,cache=True)
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


@jit(nopython=True,nogil=True,cache=True)
def eval( x, M, Y, sym=False ):
    """
    Evaluate an array of multigaussian models.
    :param x: a (n,) array of x-values to evaluate gaussians at.
    :param M: a (m,b) array containing b model parameters (where b = 3*nfeatures for symmetric and 4*nfeatures for assymetric models).
    :param Y: a (m,n) array to put results into.
    :param sym: True if M represents a set of symmetric features (3-parameter), False (Default) if it contains asymmetric (4-parameter) ones.
    """

    # evaluate symmetric model
    if sym:
        for i in range(len(M)):
            mgauss( x, Y[i,:], M[i,::3], M[i,1::3], M[i,2::3] )
    else: # evaluate asymmetric model
        for i in range(len(M)):
            amgauss( x, Y[i,:], M[i,::4], M[i,1::4], M[i,2::4], M[i,3::4] )

#########################################################
## Jacobians associated with multi-gauss functions
#########################################################

@jit(nopython=True,nogil=True,cache=True)
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
                J[i, 4 * j] = math.exp(-(x[i] - b[j]) ** 2 / c1[j])
                J[i, 4 * j + 1] = ((2 * a[j] * (x[i] - b[j])) / c1[j]) * J[i, 4 * j]
                J[i, 4 * j + 2] = ((a[j] * (x[i] - b[j]) ** 2) / c1[j] ** 2) * J[i, 4 * j]
                J[i, 4 * j + 3] = 0
            else:
                J[i, 4 * j] = math.exp(-(x[i] - b[j]) ** 2 / c2[j])
                J[i, 4 * j + 1] = ((2 * a[j] * (x[i] - b[j])) / c2[j]) * J[i, 4 * j]
                J[i, 4 * j + 2] = 0
                J[i, 4 * j + 3] = ((a[j] * (x[i] - b[j]) ** 2) / c2[j] ** 2) * J[i, 4 * j]


@jit(nopython=True,nogil=True,cache=True)
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
            J[i, j * 3] = math.exp(-(x[i] - b[j]) ** 2 / c[j])  # -e^(-(x-b)^2/c)
            J[i, j * 3 + 1] = (2 * a[j] * (x[i] - b[j]) * J[i, j * 3]) / c[j]  # ((2 a (-b + x)) e^(-(-b + x)^2/c) ) / c
            J[i, j * 3 + 2] = (a[j] * ((x[i] - b[j]) ** 2) * J[i, j * 3]) / c[j] ** 2  # (a (-b + x)^2 e^(-(x-b)^2/c) ) / c^2

#################################
## Initialization routines
#################################

@jit(nopython=True,nogil=True,cache=True)
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


@jit(nopython=True, parallel=True)
def init(x, X, n, sym=True, d=10, nthreads=-1):
    """
    Compute initial estimates of gaussian positions, widths and heights for a vector of spectra / signals.

    *Arguments*:
     :param x:  = a (n,) array containing the x-values of the spectra / signals.
     :param X: = a (m,n) array containing corresponding y-values for m different spectra / signals.
     :param n: = the number of gaussians to fit.
     :param sym: = True if symmetric gaussians should be fit.
     :param d: = the distance to test for local maxima where X[i,j±range(1,d)] <  X[i,d].
     :param nthreads: the number of threads to use for evaluation. Default is #CPUs - 1.
     :return: x1: an array of estimated gaussian functions based on local peak detection.
              See gfit.util.split_coeff( ... ) to split this into individual parameters and gfit(...) to optimize it.
    """

    # init output array
    if sym:
        out = np.zeros((X.shape[0], n * 3))  # scale, position, width for each feature
    else:
        out = np.zeros((X.shape[0], n * 4))  # scale, position, width_L, width_R for each feature

    # setup multithreading
    if nthreads != -1:  # -1 uses numba default
        t = numba.get_num_threads()  # store so we set this back later
        numba.set_num_threads(nthreads)

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
    if nthreads != -1:
        numba.set_num_threads(t)

    # return
    return out

#################################
## Least squares fitting
#################################
@jit(nopython=True,nogil=True,cache=True)
def lsq_amg(params, x, y, m, J ):
    """
    Calculate residual for least squares optimization of an asymmetric multigauss function.
    """
    # evaluate function
    amgauss(x, m, params[::4],params[1::4],params[2::4],params[3::4])
    return m - y # return residual

@jit(nopython=True,nogil=True,cache=True)
def lsq_mg(params, x, y, m, J ):
    """
    Calculate residual for least squares optimization of a symmetric multigauss function.
    """
    mgauss(x, m, params[::3],params[1::3],params[2::3])
    return m - y

@jit(nopython=True,nogil=True,cache=True)
def lsq_Jamg(params, x, y, m, J ):
    """
    Calculate and return jacobian for least-squares optimization of asymmetric multigauss function.
    """
    amgauss_J(x, J, params[::4],params[1::4],params[2::4],params[3::4])
    return J

@jit(nopython=True,nogil=True,cache=True)
def lsq_Jmg(params, x, y, m, J ):
    """
    Calculate and return jacobian for least-squares optimization of symmetric multigauss function.
    """
    mgauss_J(x, J, params[::3],params[1::3],params[2::3])
    return J

def fit_mgauss(x, y, x0, n, c=(-np.inf, np.inf), thresh=-1, ftol=1e-4, xtol=1e-4, scale=0, maxiter=100, verbose=False):
    """
    Fit a symmetric multigauss to the provided 1-D data.

    *Arguments*:
     - x = x values of data to fit.
     - y = y values of data to fit.
     - x0 = the initial guess as a stacked (1d) array [see stack_coeff].
     - n = number of gaussians to fit.
     - c = a tuple of contstrains like ([x00min,x01min,...,x0nmin],[x00max,x01max,...,x0nmax]). Set as None to use no constraints.
     - ftol = ftol parameter of scipy.optimize.least_squares. Default is 1e-4.
     - xtol = xtol parameter of scipy.optimize.least_squares. Default is 1e-4.
     - scale = scale parameter of scipy.optimize.least_squares. Default is x[1] - x[0].
     - maxiter = the maximum number of iterations allowed per fitting step. Default is 100.
    """

    # compute defaults if needed
    if scale <= 0:
        scale = x[1] - x[0]

    # check bounds
    if not check_bounds( x0, c[0], c[1] ):
        print("\nInvalid bounds or initial guess:")
        print("   x0 = %s" % x0)
        print("   c = %s" % c)

    # check threshold
    if thresh > 0:
        if np.max( x0[::3] ) < thresh:
            return x0 # don't fit

    # refine inital guess using subset of data
    m = np.zeros_like(x)  # models will be evaluated here [ to save lots of memory allocations ]
    J = np.zeros((len(x), n * 3))
    fit = least_squares(lsq_mg, x0=x0, args=(x, y, m, J), jac=lsq_Jmg,
                        x_scale=scale, bounds=c, verbose=verbose,
                        ftol=ftol, xtol=xtol, max_nfev=maxiter)
    return fit.x


def fit_amgauss(x, y, x0, n, c=(-np.inf, np.inf), thresh=-1, ftol=1e-4, xtol=1e-4, scale=0, maxiter=100, verbose=False):
    """
    Fit a asymmetric multigauss to the provided 1-D data.

    *Arguments*:
     - x = x values of data to fit.
     - y = y values of data to fit.
     - x0 = the initial guess as a stacked (1d) array [see stack_coeff].
     - n = number of gaussians to fit.
     - c = a tuple of contstrains like ([x00min,x01min,...,x0nmin],[x00max,x01max,...,x0nmax]). Set as None to use no constraints.
     - ftol = ftol parameter of scipy.optimize.least_squares. Default is 1e-4.
     - xtol = xtol parameter of scipy.optimize.least_squares. Default is 1e-4.
     - scale = scale parameter of scipy.optimize.least_squares. Default is x[1] - x[0].
     - maxiter = the maximum number of iterations allowed per fitting step. Default is 100.
    """

    # compute defaults if needed
    if scale <= 0:
        scale = x[1] - x[0]

    # check bounds
    if not check_bounds( x0, c[0,:], c[1,:] ):
        print("\nInvalid bounds or initial guess:")
        print("   x0 = %s" % x0)
        print("   c = %s" % c)

    # check threshold
    if thresh > 0:
        if np.max( x0[::4] ) < thresh:
            return x0 # don't fit

    # refine inital guess using subset of data
    m = np.zeros_like(x)  # models will be evaluated here [ to save lots of memory allocations ]
    J = np.zeros((len(x), n * 4))
    fit = least_squares(lsq_amg, x0=x0, args=(x, y, m, J), jac=lsq_Jamg,
                        x_scale=scale, bounds=c, verbose=verbose,
                        ftol=ftol, xtol=xtol, max_nfev=maxiter)
    return fit.x

@jit(nopython=True,nogil=True,cache=True)
def check_bounds( x0, mn, mx):
    """
    Checks that bounds are valid, and asserts False with a usable warning message if not.
    """
    for i in range(len(x0)):
        if not (mn[i] <= x0[i] and mx[i] >= x0[i]):
            return False
    return True

def gfit_single(x, X, x0, n, sym=True, thresh=-1, vb=True, **kwds ):
    """ Single-threaded multigaussian fitting"""

    # set number of threads to 1
    t = numba.get_num_threads()  # store so we set this back later
    numba.set_num_threads(1)

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
        out[i,:] = _opt(x,X[i,:],x0[i,:],n,c,thresh,**kwds)

    # reset number of threads
    numba.set_num_threads(t)

    # return
    return out


def _mp_opt_sym(x, X, x0, out, c0, c1, thresh, n, start, end, kwds, vb):
    """
    Multiprocessing worker function.
    """

    X, sX = X
    x0, sx0 = x0
    out, sout = out
    c = np.array([c0, c1])

    # set number of threads to use by numba
    t = numba.get_num_threads()  # store so we set this back later
    numba.set_num_threads(1)

    # do main loop
    loop = range(start, end)
    if vb:
        loop = tqdm(loop, desc='Fitting gaussians', leave=False)
    for i in loop:
        out[i * sout: (i + 1) * sout] = fit_mgauss(np.frombuffer(x, dtype=np.double),  # x-vals
                                                   np.frombuffer(X, dtype=np.double)[i * sX:(i + 1) * sX],  # y-vals
                                                   np.frombuffer(x0, dtype=np.double)[i * sx0:(i + 1) * sx0],
                                                   # initial guess
                                                   n, c, thresh, **kwds)  # number of features, number of constraints
    numba.set_num_threads(t)

def _mp_opt_asym(x, X, x0, out, c0, c1, thresh, n, start, end, kwds, vb):
    """
    Multiprocessing worker function.
    """

    # parse arrays and strides
    X, sX = X
    x0, sx0 = x0
    out, sout = out
    c = np.array([c0, c1])

    # set number of threads to use by numba
    t = numba.get_num_threads()  # store so we set this back later
    numba.set_num_threads(1)

    # do main loop
    loop = range(start, end)
    if vb:
        loop = tqdm(loop, desc='Fitting gaussians', leave=False)
    for i in loop:
        out[i * sout: (i + 1) * sout] = fit_amgauss(np.frombuffer(x, dtype=np.double),  # x-vals
                                                    np.frombuffer(X, dtype=np.double)[i * sX:(i + 1) * sX],  # y-vals
                                                    np.frombuffer(x0, dtype=np.double)[i * sx0:(i + 1) * sx0],
                                                    # initial guess
                                                    n, c, thresh, **kwds)  # number of features, number of constraints
    numba.set_num_threads(t)

def gfit_multi(x, X, x0, n, sym=True, thresh=-1, nthreads=-1, vb=True, **kwds):
    """ Single-threaded multigaussian fitting"""

    import multiprocessing as mp

    # wrap X and x0 if needed
    if len(X.shape) == 1:
        X = np.array([X])
    if len(x0.shape) == 1:
        x0 = np.array([x0])

    # reshape
    outshape = x0.shape
    X = X.reshape((-1, X.shape[-1])).astype(np.double)
    x0 = x0.reshape((-1, x0.shape[-1])).astype(np.double)

    # get bounds
    c = kwds.pop("c", get_bounds(x, x0, sym=sym))

    # create shared memory for input / output arrays
    mp_x = mp.Array("d", x, lock=False)
    mp_X = (mp.Array("d", X.ravel(order='C'), lock=False), X.shape[-1])  # N.B. these will be ( flat array, stride )
    mp_x0 = (mp.Array("d", x0.ravel(order='C'), lock=False), x0.shape[-1])  # N.B. these will be ( flat array, stride )
    mp_out = (mp.Array("d", x0.ravel(order='C'), lock=False), x0.shape[-1])
    mp_c0 = mp.Array("d", np.array(c[0]), lock=False)
    mp_c1 = mp.Array("d", np.array(c[1]), lock=False)

    # build and launch threads
    if nthreads == -1:
        nthreads = mp.cpu_count() - 1
    proc = []
    idx = [(int(X.shape[0] / nthreads) * i, int(X.shape[0] / nthreads) * (i + 1)) for i in range(nthreads - 1)]
    for start, end in idx:
        if sym:
            p = mp.Process(target=_mp_opt_sym, args=(mp_x, mp_X, mp_x0, mp_out, mp_c0, mp_c1, thresh, n, start, end - 1, kwds, False))
        else:
            p = mp.Process(target=_mp_opt_asym,
                           args=(mp_x, mp_X, mp_x0, mp_out, mp_c0, mp_c1, thresh, n, start, end - 1, kwds, False))
        p.start()
        proc.append(p)

    # crunch our own data in this thread (largely to display a meaningful progress bar)
    if sym:
        _mp_opt_sym(mp_x, mp_X, mp_x0, mp_out, mp_c0, mp_c1, thresh, n, idx[-1][-1], X.shape[0], kwds, vb)
    else:
        _mp_opt_asym(mp_x, mp_X, mp_x0, mp_out, mp_c0, mp_c1, thresh, n, idx[-1][-1], X.shape[0], kwds, vb)

    # wait until threads are complete
    for p in proc:
        p.join()

    # return results
    return np.array(mp_out[0]).reshape(outshape)


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
