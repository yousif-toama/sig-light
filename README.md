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

# Backpropagation
deriv = np.ones_like(signature)
grad = sig_light.sigbackprop(deriv, path, 3)  # gradient w.r.t. path

# Batching: process multiple paths at once
paths = np.random.randn(10, 50, 2)  # 10 paths, 50 points, 2D
sigs = sig_light.sig(paths, 3)  # shape (10, siglength(2, 3))
```

## API Reference

### Signature

#### `sig(path, m, format=0)`

Compute the signature of a path truncated at depth `m`.

- **path**: numpy array of shape `(..., n, d)`. Extra leading dims are batched.
- **m**: truncation depth (positive integer).
- **format**: output format.
  - `0`: flat array of shape `(..., siglength(d, m))`.
  - `1`: list of `m` arrays, one per level.
  - `2`: cumulative prefix signatures, shape `(..., n-1, siglength(d, m))`.
- **Returns**: the path signature, excluding the level-0 term (always 1).

#### `siglength(d, m)`

Length of the signature output: `d + d^2 + ... + d^m`.

#### `sigcombine(sig1, sig2, d, m)`

Combine two signatures via Chen's identity. Supports batching.

### Backpropagation

#### `sigbackprop(deriv, path, m)`

Compute gradient of a loss w.r.t. the path, given gradient w.r.t. the signature.

- **deriv**: shape `(..., siglength(d, m))`.
- **path**: shape `(..., n, d)`.
- **Returns**: shape `(..., n, d)`.

#### `sigjacobian(path, m)`

Full Jacobian matrix of `sig()` w.r.t. the path.

- **Returns**: shape `(n, d, siglength(d, m))`.

#### `logsigbackprop(deriv, path, s)`

Compute gradient of a loss w.r.t. the path, given gradient w.r.t. the log signature.

- **deriv**: shape `(..., logsiglength(d, m))`.
- **path**: shape `(..., n, d)`.
- **s**: prepared data from `prepare(d, m)`.
- **Returns**: shape `(..., n, d)`.

### Log Signature

#### `prepare(d, m)`

Precompute data for log signature computation.

#### `logsig(path, s)`

Compute the log signature in the Lyndon basis. Supports batching.

#### `logsig_expanded(path, s)`

Compute the log signature in the full tensor expansion. Supports batching.

#### `logsiglength(d, m)`

Length of the log signature output (Witt's formula).

#### `basis(s)`

Get the Lyndon bracket labels for the log signature basis elements.

### Transforms

#### `sigjoin(sig, segment, d, m, fixedLast=nan)`

Extend a signature by appending a linear segment. Equivalent to `sigcombine(sig, sig_of_segment(segment), d, m)`.

#### `sigjoinbackprop(deriv, sig, segment, d, m, fixedLast=nan)`

Gradient through `sigjoin`. Returns `(dsig, dsegment)` or `(dsig, dsegment, dfixedLast)`.

#### `sigscale(sig, scales, d, m)`

Rescale a signature as if each path dimension were multiplied by a factor. At level k, the multi-index `(i1,...,ik)` component is multiplied by `scales[i1] * ... * scales[ik]`.

#### `sigscalebackprop(deriv, sig, scales, d, m)`

Gradient through `sigscale`. Returns `(dsig, dscales)`.

### Rotation Invariants (2D paths)

#### `rotinv2dprepare(m, type="a")`

Precompute rotation-invariant features for 2D paths.

- **m**: depth (should be even).
- **type**: `"a"` for all invariants.

#### `rotinv2d(path, s)`

Compute rotation-invariant features of a 2D path signature.

#### `rotinv2dlength(s)` / `rotinv2dcoeffs(s)`

Get the number of invariants and their coefficient matrices.

### Utility

#### `version()`

Return the sig-light version string.

## Algorithm

sig-light uses the standard approach for computing signatures of piecewise-linear paths:

1. **Segment signature**: for each linear segment with displacement `h`, the signature is the truncated exponential `exp(h) = 1 + h + h^2/2! + h^3/3! + ...` in the tensor algebra.

2. **Chen's identity**: the signature of a concatenated path equals the tensor product of the individual segment signatures: `S(path) = S(seg_1) * S(seg_2) * ... * S(seg_n)`.

3. **Log signature**: computed by taking the tensor logarithm `log(S) = (S-1) - (S-1)^2/2 + (S-1)^3/3 - ...` and projecting onto the Lyndon word basis.

4. **Backpropagation**: reverse-mode differentiation through the Chen's identity chain using adjoint operations on the tensor algebra.

This approach is exact for piecewise-linear paths (no numerical approximation from ODE solvers).

## Comparison with iisignature

| Feature | sig-light | iisignature |
|---|---|---|
| Language | Pure Python + numpy | C++ with Python bindings |
| Installation | `pip install` (no compilation) | Requires C++ compiler |
| Performance | Slower (~7-25x for sig) | Faster |
| API | Full parity | Full |
| Backpropagation | Yes | Yes |
| Batching | Yes | Yes |
| Rotation invariants | Yes (type "a") | Yes (types a/s/q/k) |

sig-light implements the full iisignature API with an identical interface. It trades performance for portability and simplicity.

## Development

Requires [just](https://github.com/casey/just) for task running.

```bash
# Clone and install
git clone https://github.com/yousif-toama/sig-light.git
cd sig-light
just sync

# Run tests
just test

# Code quality (lint + format check + type check)
just check

# Run tests with coverage
just test-cov

# Run benchmark
just bench

# Install iisignature for cross-validation testing (requires C++ compiler)
uv pip install setuptools
uv pip install iisignature --no-build-isolation
uv run pytest tests/test_api_compat.py
```

## License

MIT
