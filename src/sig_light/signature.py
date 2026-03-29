"""Path signature computation via Chen's identity.

The signature of a piecewise-linear path is computed exactly by:
1. Computing the truncated exponential for each linear segment.
2. Composing segment signatures via Chen's identity (tensor multiplication).
"""

from typing import Literal, overload

import numpy as np
from numpy.typing import NDArray

from sig_light.algebra import (
    concat_levels,
    sig_of_segment_batch,
    split_signature,
    tensor_multiply,
    tensor_multiply_batch,
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


@overload
def sig(
    path: NDArray[np.float64],
    m: int,
    format: Literal[0] = ...,
) -> NDArray[np.float64]: ...


@overload
def sig(
    path: NDArray[np.float64],
    m: int,
    format: Literal[1] = ...,
) -> list[NDArray[np.float64]]: ...


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

    Uses batched segment signatures and binary tree reduction for
    performance. All segment signatures are computed in parallel,
    then combined pairwise using batched tensor multiplication.

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

    # Batch compute all segment signatures at once
    displacements = np.diff(path, axis=0)  # (n-1, d)
    levels = sig_of_segment_batch(displacements, m)

    # Binary tree reduction: combine pairs at each layer
    # Preserves left-to-right order via associativity
    while levels[0].shape[0] > 1:
        batch = levels[0].shape[0]

        if batch % 2 == 1:
            remainder = [lev[-1:] for lev in levels]
            levels = [lev[:-1] for lev in levels]
        else:
            remainder = None

        left = [lev[0::2] for lev in levels]
        right = [lev[1::2] for lev in levels]
        levels = tensor_multiply_batch(left, right)

        if remainder is not None:
            levels = [np.concatenate([lev, rem]) for lev, rem in zip(levels, remainder)]

    return [lev[0] for lev in levels]


def _zeros_by_level(
    d: int,
    m: int,
) -> list[NDArray[np.float64]]:
    """Create a zero signature as a list of per-level arrays."""
    return [np.zeros(d**k) for k in range(1, m + 1)]
