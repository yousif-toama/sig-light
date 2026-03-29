"""Truncated tensor algebra operations for path signature computation.

All operations use the "level-list" representation: a list of m numpy arrays
where ``levels[i]`` has shape ``(d^(i+1),)`` for i in 0..m-1.

Two conventions exist for the implicit level-0 scalar:
- "unit" (implicit 1): used by tensor_multiply (Chen's identity)
- "nil" (implicit 0): used by tensor_multiply_nil (for tensor log/exp powers)
"""

import numpy as np
from numpy.typing import NDArray


def tensor_multiply(
    a: list[NDArray[np.float64]],
    b: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    """Concatenation product in the truncated tensor algebra (implicit 1 at level 0).

    Computes the product of two elements (1 + a) and (1 + b) in the truncated
    tensor algebra, returning the non-unit part of the result.

    At each combined level k (0-indexed):
        result[k] = a[k] + b[k] + sum_{i+j=k, i>=0, j>=0} outer(a[i], b[j])

    Args:
        a: Level-list of first element (implicit 1 at level 0).
        b: Level-list of second element (implicit 1 at level 0).

    Returns:
        Level-list of the product (implicit 1 at level 0).
    """
    m = len(a)
    result = [np.copy(a[k]) + b[k] for k in range(m)]

    for k in range(m):
        for i in range(k + 1):
            j = k - 1 - i
            if j < 0:
                # i contributes with implicit 1 from b at level 0
                continue
            result[k] += np.outer(a[i], b[j]).ravel()

    return result


def tensor_multiply_nil(
    a: list[NDArray[np.float64]],
    b: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    """Concatenation product with implicit 0 at level 0 (no identity terms).

    Used for computing powers of (S - 1) in the tensor logarithm.

    At each combined level k (0-indexed):
        result[k] = sum_{i+j=k, i>=0, j>=0} outer(a[i], b[j])

    Args:
        a: Level-list (implicit 0 at level 0).
        b: Level-list (implicit 0 at level 0).

    Returns:
        Level-list of the product (implicit 0 at level 0).
    """
    m = len(a)
    result = [np.zeros_like(a[k]) for k in range(m)]

    for k in range(m):
        for i in range(k + 1):
            j = k - 1 - i
            if j < 0:
                continue
            result[k] += np.outer(a[i], b[j]).ravel()

    return result


def tensor_log(
    levels: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    """Logarithm in the truncated tensor algebra.

    Given x representing (1 + x) in the tensor algebra, computes
    log(1 + x) = x - x^2/2 + x^3/3 - x^4/4 + ...

    where powers use tensor_multiply_nil (implicit 0 at level 0).

    Args:
        levels: Level-list representing x (the non-unit part of a group-like
            element).

    Returns:
        Level-list of log(1 + x).
    """
    m = len(levels)
    # powers[n] = x^(n+1) using nil multiplication
    powers = [levels]
    for n in range(1, m):
        powers.append(tensor_multiply_nil(levels, powers[-1]))

    result = [np.copy(levels[k]) for k in range(m)]
    for n in range(1, m):
        sign = (-1) ** n
        coeff = sign / (n + 1)
        for k in range(m):
            result[k] += coeff * powers[n][k]

    return result


def sig_of_segment(
    displacement: NDArray[np.float64],
    m: int,
) -> list[NDArray[np.float64]]:
    """Signature of a single linear path segment (truncated exponential).

    For a linear path with displacement h, the signature levels are:
        level k = h^{tensor k} / k!

    Args:
        displacement: 1D array of shape (d,), the path increment.
        m: Truncation depth (number of levels to compute).

    Returns:
        Level-list of length m, where levels[i] has shape (d^(i+1),).
    """
    levels: list[NDArray[np.float64]] = [displacement.copy()]
    for k in range(2, m + 1):
        prev = levels[-1]
        levels.append(np.outer(prev, displacement).ravel() / k)
    return levels


def split_signature(
    flat_sig: NDArray[np.float64],
    d: int,
    m: int,
) -> list[NDArray[np.float64]]:
    """Split a flat signature array into a level-list.

    Args:
        flat_sig: 1D array of length siglength(d, m).
        d: Path dimension.
        m: Signature depth.

    Returns:
        Level-list of m arrays, where levels[k] has shape (d^(k+1),).
    """
    levels: list[NDArray[np.float64]] = []
    offset = 0
    for k in range(1, m + 1):
        size = d**k
        levels.append(flat_sig[offset : offset + size].copy())
        offset += size
    return levels


def concat_levels(levels: list[NDArray[np.float64]]) -> NDArray[np.float64]:
    """Concatenate a level-list into a flat signature array.

    Args:
        levels: Level-list of m arrays.

    Returns:
        1D array of length sum(d^k for k in 1..m).
    """
    return np.concatenate(levels)
