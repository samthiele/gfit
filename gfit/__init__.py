import os

# disable numpy multi-threading; we handle this.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np

from .internal import amgauss, mgauss, est_peaks, gfit_single, gfit_multi, init


#################################
## Fitting routing
#################################
def gfit(x, X, x0, n, sym=True, thresh=-1, nthreads=-1, vb=True ):
   """
   Fit multi-gaussian functions to a data array.

   :param x: a (j,) shaped array containg x-coordinates of function.
   :param X: a (i,j) shaped array containing i independent data/spectra to fit.
   :param x0: a (i,n*3) [ sym = True ] or (i,n*4) [ sym = False ] array containing initial values for optimization routine.
              See initialise( ... ) to construct this.
   :param n: the number of gaussians to fit.
   :param sym: true if gaussians are symmetric, False if they are assymetric.
   :param thresh: threshold depth to attempt fitting. If x0 has a depth < this value then fitting will not be done. A value of -1 (default) will fit all features regardless.
   :param nthreads: the number of threads to use for evaluation. Default is #CPUs - 1.
   :param vb: true if a progress bar should be created. Default is True.
   :keywords : keywords are passed directly to fit_mgauss or fit_amgauss, and can include numerical settings like ftol, xtol or maxiter.
   :return: x1: an array of optimised gaussian fuctions. See gfit.util.split_coeff( ... ) to split this into individual parameters.
   """

   if nthreads == 1:
       return gfit_single( x.astype(float), X.astype(float), x0.astype(float), n, sym=sym, thresh=thresh, vb=vb  ) # single-threaded
   else:
       return gfit_multi(x.astype(float), X.astype(float), x0.astype(float), n, sym=sym, thresh=thresh, nthreads=nthreads, vb=vb ) # multi-threaded

#################################
## Initialization routine
#################################

def initialise(x, X, n, sym=True, d=10, nthreads=-1):
    return init(x.astype(float), X.astype(float), n, sym, d, nthreads )