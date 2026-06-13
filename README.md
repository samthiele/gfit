# gfit

Fast multi-Gaussian curve fitting for spectra and other 1D signals.

`gfit` fits many independent signals at once by decomposing each into a sum of symmetric or asymmetric Gaussian peaks. It is aimed at hyperspectral and spectroscopy workflows, but works for any regularly sampled 1D data.

## Features

- Fit **symmetric** (amplitude, centre, width) or **asymmetric** (separate left/right widths) Gaussians
- Batch fitting over `(n_spectra, n_points)` arrays
- Automatic peak-based initialisation via `initialise()`
- Optional **Numba** JIT acceleration (default install)
- Pure-Python fallback when Numba is unavailable
- Multiprocessing for large batches; automatic single-threaded fallback in Pyodide / WebAssembly

## Installation

```bash
pip install gfit
```

Numba is included by default for best performance.

To install **without** Numba (e.g. Pyodide, minimal environments):

```bash
pip install gfit[nonumba]
```

Requirements: Python ≥ 3.6, NumPy, SciPy, tqdm.

## Quick start

```python
import numpy as np
from gfit import gfit, initialise, evaluate
from gfit.util import split_coeff, rand_signal

# x-axis and a batch of noisy spectra (n_spectra, n_points)
x = np.linspace(-10, 10, 200)
X = np.array([rand_signal(x, snr=14)[0] for _ in range(100)])

# Fit 3 symmetric Gaussians per spectrum
n = 3
x0 = initialise(x, X, n, sym=True, d=4)
params = gfit(x, X, x0, n, sym=True, nthreads=-1, vb=True)

# Split fitted parameters into amplitude, centre, width arrays
a, b, c = split_coeff(params, sym=True)

# Evaluate the fitted model for one spectrum
y_model = evaluate(x, params[0:1], sym=True)[0]
```

## API overview

| Function | Description |
|----------|-------------|
| `gfit(x, X, ...)` | Fit Gaussians to a batch of spectra |
| `initialise(x, X, n, ...)` | Estimate starting parameters from local peaks |
| `evaluate(x, M, sym=...)` | Evaluate one or more multi-Gaussian models |

### `gfit` parameters

| Parameter | Default | Meaning |
|-----------|---------|---------|
| `x` | — | 1D array of sample positions |
| `X` | — | 2D array `(n_spectra, n_points)` of values to fit |
| `x0` | `None` | Initial parameters; computed via `initialise()` if omitted |
| `n` | `3` | Number of Gaussian components per spectrum |
| `sym` | `True` | Symmetric (`True`) or asymmetric (`False`) peaks |
| `thresh` | `-1` | Skip fitting if peak amplitude below threshold; `-1` fits all |
| `nthreads` | `-1` | Parallel workers; `1` forces single-threaded |
| `vb` | `True` | Show tqdm progress bar |

Additional keyword arguments (`ftol`, `xtol`, `maxiter`, …) are passed to `scipy.optimize.least_squares`.

### Parameter layout

Fitted coefficients are stored as a flat array per spectrum:

- **Symmetric:** `[a₀, b₀, c₀, a₁, b₁, c₁, …]` — amplitude, centre, width
- **Asymmetric:** `[a₀, b₀, c₁₀, c₂₀, …]` — amplitude, centre, left width, right width

Use `gfit.util.split_coeff()` and `stack_coeff()` to convert between flat arrays and separate parameter lists.

## Utilities

```python
from gfit.util import remove_hull, get_bounds, benchmark

# Remove a convex hull baseline before fitting
X_corrected = remove_hull(X, upper=True, div=True)

# Bounds derived from an initial guess (also used internally by gfit)
bounds = get_bounds(x, x0, sym=True)

# Simple performance check
benchmark(size=100, res=100, nthreads=1)
```

Lower-level single-spectrum fitters are available in `gfit.internal`:

```python
from gfit.internal import fit_mgauss, fit_amgauss
```

## Performance

- With **Numba** (default): hot loops are JIT-compiled; batch initialisation can use parallel `prange`.
- With **`gfit[nonumba]`** or a missing Numba install: the same code runs as pure Python (correct, but slower). A warning is emitted once at import.
- **`nthreads=-1`** (default) uses multiprocessing to fit spectra in parallel on desktop Python.
- **`nthreads=1`** uses a single process (recommended for small batches or constrained environments).

## License

MIT — see [LICENSE](LICENSE).

## Author

Sam Thiele — [github.com/samthiele/gfit](https://github.com/samthiele/gfit)
