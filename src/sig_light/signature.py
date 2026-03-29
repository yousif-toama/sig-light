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
    sig_of_segment,
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


@overload
def sig(
    path: NDArray[np.float64],
    m: int,
    format: Literal[2] = ...,
) -> NDArray[np.float64]: ...


def sig(
    path: NDArray[np.float64],
    m: int,
    format: int = 0,
) -> NDArray[np.float64] | list[NDArray[np.float64]]:
    """Compute the signature of a path truncated at depth m.

    Uses Chen's identity: the signature of a concatenation of paths equals
    the tensor product of their individual signatures.

    Args:
        path: Array of shape (..., n, d) representing a d-dimensional path
            with n points. Extra leading dimensions are batched.
        m: Truncation depth (positive integer).
        format: Output format.
            0: flat array of shape (..., siglength(d, m)).
            1: list of m arrays, one per level (no batching support).
            2: cumulative prefix signatures, shape (..., n-1, siglength(d, m)).

    Returns:
        The path signature, excluding the level-0 term (which is always 1).
    """
    path = np.asarray(path, dtype=np.float64)

    if path.ndim > 2 and format != 1:
        return _sig_batched(path, m, format)

    if format == 2:
        return _sig_cumulative(path, m)

    levels = sig_levels(path, m)
    if format == 1:
        return levels
    return concat_levels(levels)


def _sig_batched(
    path: NDArray[np.float64],
    m: int,
    format: int,
) -> NDArray[np.float64]:
    """Handle batched paths with extra leading dimensions."""
    batch_shape = path.shape[:-2]
    n, d = path.shape[-2], path.shape[-1]
    flat_batch = path.reshape(-1, n, d)
    batch_size = flat_batch.shape[0]

    if format == 2:
        results = np.stack(
            [_sig_cumulative(flat_batch[i], m) for i in range(batch_size)]
        )
        return results.reshape(*batch_shape, n - 1, siglength(d, m))

    results = np.stack(
        [concat_levels(sig_levels(flat_batch[i], m)) for i in range(batch_size)]
    )
    return results.reshape(*batch_shape, siglength(d, m))


def _sig_cumulative(
    path: NDArray[np.float64],
    m: int,
) -> NDArray[np.float64]:
    """Compute cumulative prefix signatures (format=2).

    Returns signatures of path[:2], path[:3], ..., path[:n].

    Args:
        path: Array of shape (n, d).
        m: Truncation depth.

    Returns:
        Array of shape (n-1, siglength(d, m)).
    """
    path = np.asarray(path, dtype=np.float64)
    n, d = path.shape
    sl = siglength(d, m)

    if n < 2:
        return np.zeros((0, sl))

    displacements = np.diff(path, axis=0)
    result = np.zeros((n - 1, sl))

    # Sequential accumulation (need all intermediates)
    acc = sig_of_segment(displacements[0], m)
    result[0] = concat_levels(acc)

    for i in range(1, n - 1):
        seg = sig_of_segment(displacements[i], m)
        acc = tensor_multiply(acc, seg)
        result[i] = concat_levels(acc)

    return result


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
        sig1: Flat signature array of shape (..., siglength(d, m)).
        sig2: Flat signature array of shape (..., siglength(d, m)).
        d: Path dimension.
        m: Truncation depth.

    Returns:
        Flat signature array of shape (..., siglength(d, m)).
    """
    sig1 = np.asarray(sig1, dtype=np.float64)
    sig2 = np.asarray(sig2, dtype=np.float64)

    if sig1.ndim > 1:
        batch_shape = sig1.shape[:-1]
        flat1 = sig1.reshape(-1, sig1.shape[-1])
        flat2 = sig2.reshape(-1, sig2.shape[-1])
        results = np.stack(
            [
                _sigcombine_single(flat1[i], flat2[i], d, m)
                for i in range(flat1.shape[0])
            ]
        )
        return results.reshape(*batch_shape, siglength(d, m))

    return _sigcombine_single(sig1, sig2, d, m)


def _sigcombine_single(
    sig1: NDArray[np.float64],
    sig2: NDArray[np.float64],
    d: int,
    m: int,
) -> NDArray[np.float64]:
    """Single (non-batched) sigcombine."""
    levels1 = split_signature(sig1, d, m)
    levels2 = split_signature(sig2, d, m)
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
