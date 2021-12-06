import numpy as np
import numba
from numba import jit
from numba import prange

from .internal import amgauss, mgauss, est_peaks, gfit_single, gfit_multi


#################################
## Fitting routing
#################################
def gfit(x, X, x0, n, sym=True, nthreads=-1, vb=True ):
   """
   Fit multi-gaussian functions to a data array.

   :param x: a (j,) shaped array containg x-coordinates of function.
   :param X: a (i,j) shaped array containing i independent data/spectra to fit.
   :param x0: a (i,n*3) [ sym = True ] or (i,n*4) [ sym = False ] array containing initial values for optimization routine.
              See initialise( ... ) to construct this.
   :param n: the number of gaussians to fit.
   :param sym: true if gaussians are symmetric, False if they are assymetric.
   :param nthreads: the number of threads to use for evaluation. Default is #CPUs - 1.
   :param vb: true if a progress bar should be created. Default is True.
   :keywords : keywords are passed directly to fit_mgauss or fit_amgauss, and can include numerical settings like ftol, xtol or maxiter.
   :return: x1: an array of optimised gaussian fuctions. See gfit.util.split_coeff( ... ) to split this into individual parameters.
   """

   if nthreads == 1:
       return gfit_single( x, X, x0, n, sym=sym, vb=vb  ) # single-threaded
   else:
       return gfit_multi(x, X, x0, n, sym=sym, nthreads=nthreads, vb=vb ) # multi-threaded

#################################
## Initialization routine
#################################

@jit(nopython=True, parallel=True)
def initialise(x, X, n, sym=True, d=10, nthreads=-1):
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