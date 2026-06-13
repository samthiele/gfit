"""Tests for gfit.util without SciPy installed."""

import builtins
import os
import sys
import unittest

import numpy as np

_real_import = builtins.__import__
_patched = False


def _block_scipy(name, globals=None, locals=None, fromlist=(), level=0):
    if name == "scipy" or name.startswith("scipy."):
        raise ImportError("blocked for test")
    return _real_import(name, globals, locals, fromlist, level)


def _clear_gfit_modules():
    for name in list(sys.modules):
        if name == "gfit" or name.startswith("gfit."):
            del sys.modules[name]


def setUpModule():
    global _patched

    os.environ["GFIT_NO_NUMBA"] = "1"
    _clear_gfit_modules()

    builtins.__import__ = _block_scipy
    _patched = True

    from gfit._scipy import HAS_SCIPY

    assert not HAS_SCIPY, "noscipy tests require HAS_SCIPY to be False"


def tearDownModule():
    global _patched

    if _patched:
        builtins.__import__ = _real_import
        _patched = False

    _clear_gfit_modules()


class UtilNoScipyTestCase(unittest.TestCase):

    def test_remove_hull(self):
        from gfit.util import remove_hull

        x = np.linspace(-10, 10)
        X = np.abs(np.random.rand(10, len(x)))
        Xh = remove_hull(X, upper=True, vb=False)
        self.assertTrue(np.max(Xh) <= 1.0)

    def test_split_stack(self):
        from gfit.util import split_coeff, stack_coeff

        arr = np.array([1.0, 2.0, 3.0, 4.0, 5.0, 6.0])
        a, b, c = split_coeff(arr, sym=True)
        restored = stack_coeff(a, b, c)
        self.assertTrue(np.allclose(arr, restored))

    def test_fitting_requires_scipy(self):
        from gfit._scipy import require_least_squares

        with self.assertRaises(ImportError):
            require_least_squares()


if __name__ == "__main__":
    unittest.main()
