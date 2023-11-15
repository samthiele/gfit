import os

# disable numpy multi-threading; we handle this.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"
import numpy as np

from .internal import amgauss, mgauss, est_peaks, gfit_single, gfit_multi, init, eval


#################################
## Fitting routing
#################################
def gfit(x, X, x0=None, n=3, sym=True, thresh=-1, nthreads=-1, vb=True, **kwds ):
   """
   Fit multi-gaussian functions to a data array.

   :param x: a (j,) shaped array containg x-coordinates of function.
   :param X: a (i,j) shaped array containing i independent data/spectra to fit.
   :param x0: a (i,n*3) [ sym = True ] or (i,n*4) [ sym = False ] array containing initial values for optimization routine.
              See initialise( ... ) to construct this. If None (default) then this is constructed using initialize(...).
   :param n: the number of gaussians to fit. Default is 3.
   :param sym: true if gaussians are symmetric, False if they are assymetric.
   :param thresh: threshold depth to attempt fitting. If x0 has a depth < this value then fitting will not be done. A value of -1 (default) will fit all features regardless.
   :param nthreads: the number of threads to use for evaluation. Default is #CPUs - 1.
   :param vb: true if a progress bar should be created. Default is True.
   :keywords : keywords are passed directly to fit_mgauss or fit_amgauss, and can include numerical settings like ftol, xtol or maxiter.
   :return: x1: an array of optimised gaussian fuctions. See gfit.util.split_coeff( ... ) to split this into individual parameters.
   """
   if x0 is None:
       x0 = initialise(x,X,n,sym,nthreads=nthreads)
   if nthreads == 1:
       return gfit_single( x.astype(float), X.astype(float), x0.astype(float), n, sym=sym, thresh=thresh, vb=vb, **kwds  ) # single-threaded
   else:
       return gfit_multi(x.astype(float), X.astype(float), x0.astype(float), n, sym=sym, thresh=thresh, nthreads=nthreads, vb=vb, **kwds ) # multi-threaded

#################################
## Initialization routine
#################################

def initialise(x, X, n, sym=True, d=10, nthreads=-1):
    return init(x.astype(float), X.astype(float), n, sym, d, nthreads )

#################################
## Evaluation routine
#################################
def evaluate( x, M, sym=False ):
    """
    Evaluate an array of multigaussian models.
    :param x: a (n,) array of x-values to evaluate gaussians at.
    :param M: a (m,b) array containing b model parameters (where b = 3*nfeatures for symmetric and 4*nfeatures for assymetric models).
    :param sym: True if M represents a set of symmetric features (3-parameter), False (Default) if it contains asymmetric (4-parameter) ones.
    """

    # check shape
    if sym:
        assert M.shape[-1] % 3 == 0, "Error - %s is an invalid shape for symmetric (3-parameter) gaussians." % (M.shape)
    else:
        assert M.shape[-1] % 4 == 0, "Error - %s is an invalid shape for symmetric (3-parameter) gaussians." % (M.shape)

    # build output
    shape = M.shape[:-1] + (x.shape[0],)
    Y = np.zeros( shape, dtype=float ).reshape( (-1, x.shape[0]) )
    M = M.reshape( (-1, M.shape[-1]) )

    # evaluate
    eval(x.astype(float), M.astype(float), Y, sym)

    return Y.reshape(shape)
