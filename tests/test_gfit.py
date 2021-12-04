import unittest

import numpy as np
from gfit import amgauss, mgauss, initialise
from gfit.internal import fit_mgauss, fit_amgauss
from gfit.util import split_coeff, stack_coeff

def rand_signal(x, big=3, small=2, snr=12):
    """
    Generate a random signal for testing purposes
    :param x: x values to evaluate gaussians over
    :param big: number of big gaussian features
    :param small: number of small gaussian features
    :param snr: amount of noise
    :return: y values corresponding to the evaluated multigaussian function.
    """
    a = np.hstack([np.random.rand(big) * 2.0, np.random.rand(small) * 0.5])
    b = (np.random.rand(big + small) * np.ptp(x)) + np.min(x)  # pos
    c1 = np.random.rand(big + small) * 2 + 0.5
    c2 = np.random.rand(big + small) * 2 + 0.5

    y = np.zeros_like(x)
    amgauss(x, y, a, b, c1, c2)
    return y + (0.5 - np.random.rand(len(x))) * big / snr, a, b, c1, c2

class MyTestCase(unittest.TestCase):
    def test_forward(self):
        x = np.linspace(-10,10)
        y = np.zeros_like(x)

        # run symmetric gaussian
        mgauss(x,y, np.array([1,2,]), np.array([-2,2.5]), np.array([5,5] ) )
        self.assertTrue( (y >= 0).all() )

        # run asymmetric gaussian
        amgauss(x, y, np.array([1, 2, ]), np.array([-2, 2.5]), np.array([5, 5]), np.array([1, 1]) )
        self.assertTrue((y >= 0).all())

    def test_backward_asym(self):
        # generate test signal
        x = np.linspace(-10, 10)
        y = np.zeros_like(x)
        a = np.array([2., 1.])
        b = np.array([-5., 2.5])
        c1 = np.array([3., 5.])
        c2 = np.array([4., 2.])
        amgauss(x, y, a, b, c1, c2)  # evaluate model

        # generate initial guess
        n = 2
        x0 = initialise(x, np.array([y]), n, sym=False, d=4)  # compute initial values

        # fit it
        c = np.array(([0., -11., 0, 0] * n, [11., 11., 11., 11.] * n))
        fit = fit_amgauss(x, y, x0[0, :], n, c=c, verbose=False, ftol=1e-6, xtol=1e-6)
        fa, fb, fc1, fc2 = split_coeff(fit)

        # sort results by position and check they're similar
        idxA = np.argsort(b)  # first three features return by rand_signal are the deep ones
        idxB = np.argsort(fb)

        for i in range(n):
            self.assertTrue( np.abs(a[idxA[i]] - fa[idxB[i]]) < 0.1 )
            self.assertTrue( np.abs(b[idxA[i]] - fb[idxB[i]]) < 0.1 )
            self.assertTrue(  np.abs(c1[idxA[i]] - fc1[idxB[i]]) < 0.1 )
            self.assertTrue(  np.abs(c2[idxA[i]] - fc2[idxB[i]]) < 0.1 )

    def test_backward_sym(self):
        # generate test signal
        x = np.linspace(-10, 10)
        y = np.zeros_like(x)
        a = np.array([2., 1.])
        b = np.array([-5., 2.5])
        c1 = np.array([3., 5.])
        mgauss(x, y, a, b, c1)  # evaluate model

        # generate initial guess
        n = 2
        x0 = initialise(x, np.array([y]), n, sym=True, d=4)  # compute initial values

        # fit it
        c = np.array(([0., -11., 0] * n, [11., 11., 11.] * n))
        fit = fit_mgauss(x, y, x0[0, :], n, c=c, verbose=False, ftol=1e-6, xtol=1e-6)
        fa, fb, fc1 = split_coeff(fit, sym=True)

        # sort results by position and check they're similar
        idxA = np.argsort(b)  # first three features return by rand_signal are the deep ones
        idxB = np.argsort(fb)

        for i in range(n):
            self.assertTrue( np.abs(a[idxA[i]] - fa[idxB[i]]) < 0.1 )
            self.assertTrue( np.abs(b[idxA[i]] - fb[idxB[i]]) < 0.1 )
            self.assertTrue(  np.abs(c1[idxA[i]] - fc1[idxB[i]]) < 0.1 )

    def test_splitting(self):
        x = np.linspace(-10, 10)
        X = np.array([rand_signal(x, snr=14)[0] for i in range(10)])  # create random array
        x0 = initialise(x, X, 3, sym=False, d=4)  # compute initial values

        # check that split and stack functions work for assymetric functions
        a, b, c1, c2 = split_coeff(x0, sym=False)
        x1 = stack_coeff(a, b, c1, c2)

        assert (x0 == x1).all(), "Error - 2D stacking or splitting doesn't work"
        assert (stack_coeff(*split_coeff(x0[0, :])) == x0[0, :]).all(), "Error - 1D stacking or splitting doesn't work"

        # check that split and stack functions work for symmetric functions
        x0 = initialise(x, X, 3, sym=True, d=4)  # compute initial values
        a, b, c = split_coeff(x0, sym=True)
        x1 = stack_coeff(a, b, c)
        assert (x0 == x1).all(), "Error - 2D stacking or splitting doesn't work"

    def test_gfit_single_sym(self):
        from gfit import gfit

        x = np.linspace(-10, 10)
        X = np.array([rand_signal(x, snr=14)[0] for i in range(1000)])  # create random array

        x0 = initialise(x, X, 3, sym=True, d=4)  # compute initial values
        F = gfit(x, X, x0, 3, sym=True, nthreads=1, vb=True)


if __name__ == '__main__':
    unittest.main()
