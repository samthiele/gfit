"""Optional SciPy dependency (required only for least-squares fitting)."""

try:
    from scipy.optimize import least_squares

    HAS_SCIPY = True
except ImportError:
    least_squares = None
    HAS_SCIPY = False


def require_least_squares():
    """
    Return scipy.optimize.least_squares, or raise a clear ImportError.

    Hull correction, model evaluation, and initialisation do not need SciPy.
    """
    if HAS_SCIPY:
        return least_squares
    raise ImportError(
        "scipy is required for Gaussian fitting (gfit, fit_mgauss, fit_amgauss). "
        "Install with: pip install gfit   or   pip install scipy"
    )
