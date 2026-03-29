# CLAUDE.md

## Project Overview

sig-light is a pure-Python library for computing path signatures and log signatures of multidimensional time series. It mirrors the iisignature API using Chen's identity over piecewise-linear paths with truncated tensor algebra operations. No C extensions — numpy only.

## Key Commands

```bash
# Install dependencies
uv sync

# Run tests
uv run pytest

# Code quality — run all three before committing
uv run ruff check src/ tests/
uv run ruff format --check src/ tests/
uv run ty check
```

## Architecture

### Core Algorithm

- Signature via Chen's identity: segment signatures (truncated exp) composed via tensor multiplication
- Log signature via "S" method: compute signature, take tensor logarithm, project to Lyndon basis
- All tensor algebra operations work on "level-lists": list of m arrays, level[k] has shape (d^(k+1),)

### File Structure

```
src/sig_light/
  __init__.py       # Public API re-exports, version()
  algebra.py        # Tensor algebra: multiply, exp, log, sig_of_segment
  signature.py      # sig(), siglength(), sigcombine()
  logsignature.py   # logsig(), logsiglength(), prepare(), basis()
  lyndon.py         # Lyndon word generation, bracketing, basis construction
```

### Key Types

- **Level-list**: `list[NDArray]` of length m, representing truncated tensor algebra element (implicit 1 or 0 at level 0 depending on context)
- **PreparedData**: frozen dataclass from `prepare()`, holds precomputed Lyndon basis info
- **Flat signature**: 1D ndarray of length `siglength(d, m)`, levels 1..m concatenated

## Dependencies

- numpy >= 2.0 (core computation)
- Dev: ruff, ty, pytest, pytest-cov

## API Compatibility

Mirrors iisignature API. Signature does NOT include level 0 ("1") — starts from level 1, matching iisignature convention.
