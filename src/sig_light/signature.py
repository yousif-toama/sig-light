"""Path signature computation via Chen's identity.

The signature of a piecewise-linear path is computed exactly by:
1. Computing the truncated exponential for each linear segment.
2. Composing segment signatures via Chen's identity (tensor multiplication).
"""

import numpy as np
from numpy.typing import NDArray

from sig_light.algebra import (
    concat_levels,
    sig_of_segment,
    split_signature,
    tensor_multiply,
)


def siglength(d: int, m: int) -> int:
    """Length of the signature output (levels 1 through m).

    Args:
        d: Dimension of the path.
        m: Truncation depth.

    Returns:
        d + d^2 + ... + d^m = d*(d^m - 1)/(d - 1) for d > 1, else m.
    """
    if d == 1:
        return m
    return d * (d**m - 1) // (d - 1)


def sig(
    path: NDArray[np.float64],
    m: int,
    format: int = 0,
) -> NDArray[np.float64] | list[NDArray[np.float64]]:
    """Compute the signature of a path truncated at depth m.

    Uses Chen's identity: the signature of a concatenation of paths equals
    the tensor product of their individual signatures.

    Args:
        path: Array of shape (n, d) representing a d-dimensional path
            with n points. Must have dtype float32 or float64.
        m: Truncation depth (positive integer).
        format: Output format.
            0: flat 1D array of length siglength(d, m).
            1: list of m arrays, one per level.

    Returns:
        The path signature, excluding the level-0 term (which is always 1).
    """
    levels = sig_levels(path, m)
    if format == 1:
        return levels
    return concat_levels(levels)


def sigcombine(
    sig1: NDArray[np.float64],
    sig2: NDArray[np.float64],
    d: int,
    m: int,
) -> NDArray[np.float64]:
    """Combine two signatures via Chen's identity.

    Given signatures S1 and S2 of two paths, computes the signature of
    their concatenation: S(path1 * path2) = S1 tensor S2.

    Args:
        sig1: Flat signature array of length siglength(d, m).
        sig2: Flat signature array of length siglength(d, m).
        d: Path dimension.
        m: Truncation depth.

    Returns:
        Flat signature array of length siglength(d, m).
    """
    levels1 = split_signature(np.asarray(sig1, dtype=np.float64), d, m)
    levels2 = split_signature(np.asarray(sig2, dtype=np.float64), d, m)
    result = tensor_multiply(levels1, levels2)
    return concat_levels(result)


def sig_levels(
    path: NDArray[np.float64],
    m: int,
) -> list[NDArray[np.float64]]:
    """Compute the signature as a level-list (internal API).

    Always returns a list of per-level arrays. Used by logsignature.py
    to avoid union return type.

    Args:
        path: Array of shape (n, d).
        m: Truncation depth.

    Returns:
        Level-list of m arrays.
    """
    path = np.asarray(path, dtype=np.float64)
    n, d = path.shape

    if n < 2:
        return _zeros_by_level(d, m)

    h = path[1] - path[0]
    levels = sig_of_segment(h, m)

    for i in range(2, n):
        h = path[i] - path[i - 1]
        seg = sig_of_segment(h, m)
        levels = tensor_multiply(levels, seg)

    return levels


def _zeros_by_level(
    d: int,
    m: int,
) -> list[NDArray[np.float64]]:
    """Create a zero signature as a list of per-level arrays."""
    return [np.zeros(d**k) for k in range(1, m + 1)]
