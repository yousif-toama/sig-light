# sig-light

[![CICD](https://github.com/yousif-toama/sig-light/actions/workflows/cicd.yml/badge.svg)](https://github.com/yousif-toama/sig-light/actions/workflows/cicd.yml)
[![codecov](https://codecov.io/gh/yousif-toama/sig-light/graph/badge.svg)](https://codecov.io/gh/yousif-toama/sig-light)

Pure-Python path signature and log signature computation, mirroring the [iisignature](https://github.com/bottler/iisignature) API.

sig-light computes signatures and log signatures of multidimensional piecewise-linear paths using Chen's identity and truncated tensor algebra operations. No C extensions or compilation required — just numpy.

## Installation

```bash
pip install sig-light
```

Or with uv:

```bash
uv add sig-light
```

## Quick Start

```python
import numpy as np
import sig_light

# Define a 2D path
path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])

# Compute signature at depth 3
signature = sig_light.sig(path, 3)

# Compute log signature
s = sig_light.prepare(2, 3)
log_signature = sig_light.logsig(path, s)

# View the basis labels
print(sig_light.basis(s))
# ['1', '2', '[1,2]', '[1,[1,2]]', '[[1,2],2]']
```

## API Reference

### Signature

#### `sig(path, m, format=0)`

Compute the signature of a path truncated at depth `m`.

- **path**: numpy array of shape `(n, d)` — a `d`-dimensional path with `n` points.
- **m**: truncation depth (positive integer).
- **format**: output format.
  - `0`: flat 1D array of length `siglength(d, m)`.
  - `1`: list of `m` arrays, one per level.
- **Returns**: the path signature, excluding the level-0 term (always 1).

```python
path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])

sig_light.sig(path, 2)
# array([1. , 1. , 0.5, 1. , 0. , 0.5])

sig_light.sig(path, 2, format=1)
# [array([1., 1.]), array([0.5, 1. , 0. , 0.5])]
```

#### `siglength(d, m)`

Length of the signature output (levels 1 through `m`).

```python
sig_light.siglength(3, 3)  # 39
```

#### `sigcombine(sig1, sig2, d, m)`

Combine two signatures via Chen's identity. If `sig1` is the signature of path A and `sig2` is the signature of path B, returns the signature of their concatenation.

```python
s1 = sig_light.sig(path[:2], 2)
s2 = sig_light.sig(path[1:], 2)
combined = sig_light.sigcombine(s1, s2, 2, 2)
# Equals sig_light.sig(path, 2)
```

### Log Signature

#### `prepare(d, m)`

Precompute data for log signature computation. Returns an opaque object used by `logsig()` and `basis()`.

- **d**: path dimension.
- **m**: truncation depth.

```python
s = sig_light.prepare(2, 3)
```

#### `logsig(path, s)`

Compute the log signature of a path in the Lyndon basis.

- **path**: numpy array of shape `(n, d)`.
- **s**: prepared data from `prepare(d, m)`.
- **Returns**: 1D array of length `logsiglength(d, m)`.

```python
s = sig_light.prepare(2, 2)
sig_light.logsig(np.array([[0., 0.], [1., 0.], [1., 1.]]), s)
# array([1. , 1. , 0.5])
```

#### `logsig_expanded(path, s)`

Compute the log signature in the full tensor expansion (no Lyndon projection). Output length equals `siglength(d, m)`.

#### `logsiglength(d, m)`

Length of the log signature output (number of Lyndon words up to length `m`). Computed via Witt's formula.

```python
sig_light.logsiglength(3, 3)  # 14
```

#### `basis(s)`

Get the Lyndon bracket labels for the log signature basis elements.

```python
s = sig_light.prepare(2, 3)
sig_light.basis(s)
# ['1', '2', '[1,2]', '[1,[1,2]]', '[[1,2],2]']
```

### Utility

#### `version()`

Return the sig-light version string.

## Algorithm

sig-light uses the standard approach for computing signatures of piecewise-linear paths:

1. **Segment signature**: for each linear segment with displacement `h`, the signature is the truncated exponential `exp(h) = 1 + h + h^2/2! + h^3/3! + ...` in the tensor algebra.

2. **Chen's identity**: the signature of a concatenated path equals the tensor product of the individual segment signatures: `S(path) = S(seg_1) * S(seg_2) * ... * S(seg_n)`.

3. **Log signature**: computed by taking the tensor logarithm `log(S) = (S-1) - (S-1)^2/2 + (S-1)^3/3 - ...` and projecting onto the Lyndon word basis.

This approach is exact for piecewise-linear paths (no numerical approximation from ODE solvers).

## Comparison with iisignature

| Feature | sig-light | iisignature |
|---|---|---|
| Language | Pure Python + numpy | C++ with Python bindings |
| Installation | `pip install` (no compilation) | Requires C++ compiler |
| Performance | Slower | Faster |
| API | Compatible subset | Full |
| Backpropagation | Not yet | Yes |
| Batching | Not yet | Yes |

sig-light implements the core signature and log signature functions with an API matching iisignature. It trades performance for portability and simplicity.

## Development

```bash
# Clone and install
git clone https://github.com/yousif-toama/sig-light.git
cd sig-light
uv sync

# Run tests
uv run pytest

# Install iisignature for cross-validation testing (requires C++ compiler)
uv pip install setuptools
uv pip install iisignature --no-build-isolation
uv run pytest tests/test_api_compat.py

# Run benchmark
uv run python scripts/benchmark.py

# Code quality
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check
```

## License

MIT
