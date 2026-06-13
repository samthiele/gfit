"""
Optional numba acceleration with pure-Python fallbacks.

gfit's hot paths are written once as plain Python loops and decorated with
@jit. This module selects the backend at import time:

  - numba available  -> real @jit / prange (fast, default install)
  - numba missing    -> no-op @jit, prange=range (slower but fully functional)
  - GFIT_NO_NUMBA=1  -> force the fallback path even if numba is installed
                        (used by tests/test_gfit_nonumba.py)

internal.py and util.py import jit, prange, get_num_threads, and
set_num_threads from here rather than from numba directly.
"""

import os
import warnings

# Test hook: set GFIT_NO_NUMBA=1 (or call set_simulate_no_numba()) before
# importing gfit to exercise the pure-Python path without uninstalling numba.
_simulate_no_numba = os.environ.get("GFIT_NO_NUMBA", "").lower() in ("1", "true", "yes")
_warned_no_numba = False

# Defaults match the no-numba fallback; _configure() replaces these when
# numba is available and simulation is disabled.
numba = None
HAS_NUMBA = False
prange = range


def _noop_jit(*args, **kwargs):
    """Return the function unchanged when numba is unavailable."""
    def decorator(func):
        return func

    return decorator


jit = _noop_jit


def _warn_no_numba():
    global _warned_no_numba
    if not _warned_no_numba:
        _warned_no_numba = True
        warnings.warn(
            "numba is not installed; gfit will use slower pure Python fallbacks.",
            ImportWarning,
            stacklevel=2,
        )


def _use_fallback(warn=False):
    """Switch to pure-Python implementations (no JIT, no parallel prange)."""
    global numba, HAS_NUMBA, jit, prange
    numba = None
    HAS_NUMBA = False
    prange = range
    jit = _noop_jit
    if warn:
        _warn_no_numba()


def _use_numba():
    """Bind real numba symbols when the package is importable."""
    global numba, HAS_NUMBA, jit, prange
    import numba as _numba
    from numba import jit as _jit, prange as _prange

    numba = _numba
    HAS_NUMBA = True
    jit = _jit
    prange = _prange


def set_simulate_no_numba(enabled=True):
    """
    Force pure-Python fallbacks even when numba is installed.

    Intended for tests. After toggling, callers must reload gfit submodules
    so @jit-decorated functions in internal.py / util.py are redefined.
    """
    global _simulate_no_numba
    _simulate_no_numba = enabled
    if enabled:
        _use_fallback(warn=True)
    else:
        try:
            _use_numba()
        except ImportError:
            _use_fallback(warn=True)


def _configure():
    """Pick numba or fallback backend once at import time."""
    if _simulate_no_numba:
        _use_fallback(warn=True)
        return
    try:
        _use_numba()
    except ImportError:
        _use_fallback(warn=True)


_configure()


def get_num_threads():
    """Mirror numba.get_num_threads(); returns 1 without numba."""
    if HAS_NUMBA:
        return numba.get_num_threads()
    return 1


def set_num_threads(n):
    """Mirror numba.set_num_threads(); no-op without numba."""
    if HAS_NUMBA:
        numba.set_num_threads(n)
