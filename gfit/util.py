"""
Utility functions for e.g. stacking / splitting parameter arrays or estimating bounds.
"""

import os

# disable numpy multi-threading; we handle this.
os.environ["OMP_NUM_THREADS"] = "1"
os.environ["OPENBLAS_NUM_THREADS"] = "1"
os.environ["MKL_NUM_THREADS"] = "1"
os.environ["VECLIB_MAXIMUM_THREADS"] = "1"
os.environ["NUMEXPR_NUM_THREADS"] = "1"

import numpy as np
from numba import jit
from tqdm import tqdm
import timeit

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


##############
## Detrending
##############
@jit(nopython=True)
def _split(y, start, end, n):
    """
    Recursively find split points to resolve the convex hull.

    Returns a list of split points between start and end.
    """

    if (end-start <= 1): # special case - segment has only one data point in it
        return [start,end]
    if n > 100: # escape for max recursion [ reports and stops weird crashes occasionally ]
        print("Error - recursion depth exceeded for segment", start, " - ", end )
        print("Problematic Y values:", y[start:end])
        print("All Y values:", y)
        assert False

    # compute gradient
    dy = (y[end] - y[start]) / (end - start)

    # find max of trend-removed deltas
    midx = 0
    mv = 0  # set 0 here as we want to ignore negative deltas
    for i in range(start, end + 1):
        delta = y[i] - (y[start] + dy * (i - start))  # deviation from trend
        if delta > mv:
            mv = delta
            midx = i
    if mv <= 1e-10:  # we need a tolerance to avoid floating point errors
        return [start, end]  # this is a complete segment!
    else:
        return _split(y, start, midx, n+1)[:-1] + _split(y, midx, end, n+1)  # find inner segments and return

@jit(nopython=True)
def _remove_hull(y, upper=True, div=True):
    """
    Find convex hull and do trend removal to a 1D signal.
    Warning: this modifies y in-place.
    """
    # get split points
    if upper:
        v = _split(y, 0, int(len(y) - 1), int(0) )
    else:
        v = _split(-y, 0, int(len(y) - 1), int(0))

    # evaluate and remove trend
    for p in range(1, len(v)):  # loop through vertices in hull
        m = (y[v[p]] - y[v[p - 1]]) / (v[p] - v[p - 1])
        c = y[v[p - 1]]
        for j in range(v[p - 1], v[p]):  # evaluate segment
            if div:
                y[j] /= m * (j - v[p - 1]) + c
            else:
                y[j] -= m * (j - v[p - 1]) + c
    if div:
        y[-1] = 1.0  # add last value
    else:
        y[-1] = 0.0


def remove_hull(X, upper=True, div=True, vb=False):
    """
    Fit a convex hull to the specified data and remove it.

    :param X: = a n-d array containing signals that the convex hull will be calculated for in its last dimension.
    :param upper: = True (default) if an upper convex hull is returned (rather than a lower one).
    :param div: = True if the convex hull should be removed by division (as opposed to subtraction). Default is True.
    :param vb: = True if a progress bar should be created. Default is True.
    """

    # create output array
    s = X.shape
    X = X.reshape((-1, X.shape[-1]))
    out = X.copy().astype(float)

    # ensure all values are positive and non-zero to avoid wierd stuff when signals have + and - values.
    out -= np.min(out,axis=-1)[:,None] - 0.1

    # do trend removal
    loop = range(X.shape[0])
    if vb:
        loop = tqdm(loop, desc='Removing hull', leave=False)
    for i in loop:
        _remove_hull(out[i, :], upper=upper, div=div)  # remove hull

    # return
    return out.reshape(s)

def rand_signal(x, big=3, small=2, snr=12):
    """
    Generate a random signal for testing purposes
    :param x: x values to evaluate gaussians over
    :param big: number of big gaussian features
    :param small: number of small gaussian features
    :param snr: amount of noise
    :return: y values corresponding to the evaluated multigaussian function.
    """

    from gfit.internal import amgauss

    a = np.hstack([np.random.rand(big) * 2.0, np.random.rand(small) * 0.5])
    b = (np.random.rand(big + small) * np.ptp(x)) + np.min(x)  # pos
    c1 = np.random.rand(big + small) * 2 + 0.5
    c2 = np.random.rand(big + small) * 2 + 0.5

    y = np.zeros_like(x)
    amgauss(x, y, a, b, c1, c2)
    return y + (0.5 - np.random.rand(len(x))) * big / snr, a, b, c1, c2


def benchmark(size=1000, res=100, it=10, nf=3, nthreads=1, vb=True):
    """
    Create and fit some simple benchmark data for gauging performance.
    :param size: number of models to fit. Default is 10000.
    :param res: number of points in each spectra to fit. Default is 100.
    :param it: number of times to repeat each operation being benchmarked. Default is 10.
    :param nf: number of features to use for each operation being benchmarked. Default is 3.
    :param nthreads: number of threads to use for computation. Default is 1.
    :param vb: True if benchmark results should be printed.
    :return: a dictionary of benchmark times.
    """

    # setup output and progress bar
    B = {}
    from tqdm import tqdm
    pbar = tqdm(total=5, desc='Running benchmarks', leave=False)

    from gfit.internal import amgauss
    x = np.linspace(-100, 100, res)
    y = np.zeros_like(x)
    a = np.hstack([np.random.rand(nf) * 2.0])
    b = (np.random.rand(nf) * np.ptp(x)) + np.min(x)  # pos',
    c1 = np.random.rand(nf) * 2 + 0.5
    c2 = np.random.rand(nf) * 2 + 0.5

    # run a basic forward model benchmark
    def b1():
        amgauss(x, y, a, b, c1, c2)

    B['Multigauss evaluation'] = (timeit.timeit(b1, number=it, globals=globals()), it)
    pbar.update(1)

    # run a symmetric initialisation benchmark
    from gfit.util import rand_signal
    from gfit import initialise
    X = np.array([rand_signal(x, snr=20)[0] for i in range(size)])  # create random test dataset
    x0_sym = initialise(x, X, nf, sym=True, d=4, nthreads=nthreads)  # we use this later

    def b2():
        x0 = initialise(x, X, nf, sym=True, d=4, nthreads=nthreads)  # compute initial values

    B['Initialisation (symmetric)'] = (timeit.timeit(b2, number=it, globals=globals()), it)
    pbar.update(1)

    # run a asymmetric initialisation benchmark
    from gfit.util import rand_signal
    X = np.array([rand_signal(x, snr=14)[0] for i in range(size)])  # create random test dataset
    x0_asym = initialise(x, X, nf, sym=False, d=4, nthreads=nthreads)  # we use this later

    def b3():
        initialise(x, X, nf, sym=False, d=4, nthreads=nthreads)  # compute initial values

    B['Initialisation (asymmetric)'] = (timeit.timeit(b3, number=it, globals=globals()), it)
    pbar.update(1)

    # run a symmetric fitting benchmark
    from gfit import gfit
    def b4():
        gfit(x, X, x0_sym, nf, sym=True, nthreads=nthreads, vb=False)

    B['Fitting (symmetric)'] = (timeit.timeit(b4, number=it, globals=globals()), it)
    pbar.update(1)

    # run asymmetric fitting benchmark
    from gfit import gfit
    def b5():
        gfit(x, X, x0_asym, nf, sym=False, nthreads=nthreads, vb=False)

    B['Fitting (asymmetric)'] = (timeit.timeit(b5, number=it, globals=globals()), it)
    pbar.update(1)
    pbar.close()

    if vb:  # make a pretty print out
        print("\nBenchmark using %d signals of length %d and fitting %d features." % (size, res, nf))
        print("Using %d computation threads [Use -1 to set maximum default]." % nthreads)
        for k, v in B.items():
            t = float(v[0]) / (float(v[1])*size)
            if t < 1e-7:
                print(" - ", k, ': %.3f ns per signal' % (t * 1e9))
            elif t <1e-4:
                print(" - ", k, ': %.3f μs per signal' % (t * 1e6))
            elif t < 1e-2:  # use ms
                print(" - ", k, ': %.3f ms per signal' % (t * 1e3))
            else:
                print(" - ", k, ': %.3f seconds signal' % t)

    return B
