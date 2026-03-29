# CLAUDE.md

## Project Overview

sig-light is a pure-Python library for computing path signatures and log signatures of multidimensional time series. It mirrors the iisignature API using Chen's identity over piecewise-linear paths with truncated tensor algebra operations. No C extensions — numpy only.

## Key Commands

```bash
just sync       # Install dependencies
just test       # Run all tests
just test-cov   # Run tests with coverage
just check      # Run all code quality checks (lint + format + typecheck)
just lint       # Run ruff check
just format     # Auto-format code
just typecheck  # Run ty check
just bench      # Run benchmark
```

## Architecture

### Core Algorithm

- Signature via Chen's identity: segment signatures (truncated exp) composed via tensor multiplication
- Log signature via "S" method: compute signature, take tensor logarithm, project to Lyndon basis
- Backpropagation via adjoint operations on tensor algebra primitives
- All tensor algebra operations work on "level-lists": list of m arrays, level[k] has shape (d^(k+1),)
- Batched operations use einsum for vectorized computation across segments

### File Structure

```
src/sig_light/
  __init__.py       # Public API re-exports, version()
  algebra.py        # Tensor algebra: multiply, exp, log, adjoints, batched ops
  signature.py      # sig(), siglength(), sigcombine(), format=2, batching
  logsignature.py   # logsig(), logsiglength(), prepare(), basis(), batching
  backprop.py       # sigbackprop(), sigjacobian(), logsigbackprop()
  transforms.py     # sigjoin(), sigscale() + backprop variants
  lyndon.py         # Lyndon word generation, bracketing, basis construction
  rotational.py     # rotinv2d functions for 2D rotation-invariant features
```

### Key Types

- **Level-list**: `list[NDArray]` of length m, representing truncated tensor algebra element (implicit 1 or 0 at level 0 depending on context)
- **PreparedData**: frozen dataclass from `prepare()`, holds precomputed Lyndon basis info
- **RotInv2DPreparedData**: frozen dataclass for rotation invariant computation
- **Flat signature**: 1D ndarray of length `siglength(d, m)`, levels 1..m concatenated

## Dependencies

- numpy >= 2.0 (core computation)
- Dev: ruff, ty, pytest, pytest-cov

## API Compatibility

Mirrors iisignature API. Signature does NOT include level 0 ("1") — starts from level 1, matching iisignature convention. All public functions support batching via extra leading dimensions.
