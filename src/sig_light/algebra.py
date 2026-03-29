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
    result = [a[k] + b[k] for k in range(m)]

    if m >= 2:
        buf = np.empty(len(result[-1]))

    for k in range(m):
        for i in range(k + 1):
            j = k - 1 - i
            if j < 0:
                # i contributes with implicit 1 from b at level 0
                continue
            si, sj = len(a[i]), len(b[j])
            np.outer(a[i], b[j], out=buf[: si * sj].reshape(si, sj))
            result[k] += buf[: si * sj]

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

    if m >= 2:
        buf = np.empty(len(a[-1]))

    for k in range(m):
        for i in range(k + 1):
            j = k - 1 - i
            if j < 0:
                continue
            si, sj = len(a[i]), len(b[j])
            np.outer(a[i], b[j], out=buf[: si * sj].reshape(si, sj))
            result[k] += buf[: si * sj]

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
        new_level = np.outer(prev, displacement).ravel()
        new_level /= k
        levels.append(new_level)
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


# --- Batched operations for vectorized computation ---


def sig_of_segment_batch(
    displacements: NDArray[np.float64],
    m: int,
) -> list[NDArray[np.float64]]:
    """Signature of all linear segments at once (batched truncated exponential).

    For n displacements, computes all n segment signatures in parallel
    using einsum instead of per-segment Python loops.

    Args:
        displacements: Array of shape (n, d), one displacement per segment.
        m: Truncation depth.

    Returns:
        Level-list of m arrays, where levels[k] has shape (n, d^(k+1)).
    """
    levels: list[NDArray[np.float64]] = [displacements.copy()]
    for k in range(2, m + 1):
        prev = levels[-1]
        new_level = np.einsum("ni,nj->nij", prev, displacements)
        reshaped = new_level.reshape(prev.shape[0], -1)
        reshaped /= k
        levels.append(reshaped)
    return levels


# --- Adjoint (reverse-mode derivative) operations ---


def sig_of_segment_adjoint_batch(
    dresults: list[NDArray[np.float64]],
    displacements: NDArray[np.float64],
    m: int,
) -> NDArray[np.float64]:
    """Batched adjoint of sig_of_segment: gradient w.r.t. all displacements.

    Processes all n segments at once using einsum.

    Args:
        dresults: List of m arrays, each shape (n, d^(k+1)).
        displacements: Array of shape (n, d).
        m: Truncation depth.

    Returns:
        Gradient w.r.t. displacements, shape (n, d).
    """
    n, d = displacements.shape
    h = displacements

    # Recompute forward levels (batched)
    levels: list[NDArray[np.float64]] = [h.copy()]
    for k in range(2, m + 1):
        prev = levels[-1]
        new_level = np.einsum("ni,nj->nij", prev, h)
        reshaped = new_level.reshape(n, -1)
        reshaped /= k
        levels.append(reshaped)

    # Initialize dlevel from dresults
    dlevel = [dresults[k].copy() for k in range(m)]

    # dh accumulates gradient w.r.t. displacement
    dh = np.zeros_like(h)

    # Backprop through levels in reverse
    for k in range(m - 1, 0, -1):
        divisor = k + 1
        scaled = dlevel[k] / divisor  # (n, d^(k+1))
        size_prev = d**k
        mat = scaled.reshape(n, size_prev, d)  # (n, d^k, d)

        # dlevel[k-1] += mat @ h  →  einsum('nij,nj->ni', mat, h)
        dlevel[k - 1] += np.einsum("nij,nj->ni", mat, h)

        # dh += levels[k-1] @ mat  →  einsum('ni,nij->nj', levels[k-1], mat)
        prev_reshaped = levels[k - 1]  # (n, d^k)
        dh += np.einsum("ni,nij->nj", prev_reshaped, mat)

    # levels[0] = h, so dlevel[0] contributes directly
    dh += dlevel[0]

    return dh


def tensor_multiply_adjoint(
    dresult: list[NDArray[np.float64]],
    a: list[NDArray[np.float64]],
    b: list[NDArray[np.float64]],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Adjoint of tensor_multiply: gradients w.r.t. both inputs.

    Given gradient of loss w.r.t. output of tensor_multiply(a, b),
    computes gradients w.r.t. a and b.

    Args:
        dresult: Gradient w.r.t. output, same structure as output.
        a: First input to the forward tensor_multiply.
        b: Second input to the forward tensor_multiply.

    Returns:
        Tuple (da, db) of gradients w.r.t. a and b.
    """
    m = len(a)
    da = [np.copy(dresult[k]) for k in range(m)]
    db = [np.copy(dresult[k]) for k in range(m)]

    if m >= 2:
        buf_da = np.empty(len(a[-1]))
        buf_db = np.empty(len(b[-1]))

    for k in range(m):
        for i in range(k + 1):
            j = k - 1 - i
            if j < 0:
                continue
            size_i = len(a[i])
            size_j = len(b[j])
            dr = dresult[k].reshape(size_i, size_j)
            np.matmul(dr, b[j], out=buf_da[:size_i])
            da[i] += buf_da[:size_i]
            np.matmul(a[i], dr, out=buf_db[:size_j])
            db[j] += buf_db[:size_j]

    return da, db


def sig_of_segment_adjoint(
    dresult: list[NDArray[np.float64]],
    displacement: NDArray[np.float64],
    m: int,
) -> NDArray[np.float64]:
    """Adjoint of sig_of_segment: gradient w.r.t. displacement.

    Forward: levels[0] = h, levels[k] = outer(levels[k-1], h) / (k+1)
    where k is 1-indexed (levels list is 0-indexed, so levels[k] uses divisor k+1).

    Args:
        dresult: Gradient w.r.t. output levels.
        displacement: The input displacement vector.
        m: Truncation depth.

    Returns:
        Gradient w.r.t. displacement, shape (d,).
    """
    d = len(displacement)
    h = displacement

    # Recompute forward levels
    levels: list[NDArray[np.float64]] = [h.copy()]
    for k in range(2, m + 1):
        new_level = np.outer(levels[-1], h).ravel()
        new_level /= k
        levels.append(new_level)

    # Initialize dlevel from dresult
    dlevel = [np.copy(dresult[k]) for k in range(m)]

    # dh accumulates the total gradient w.r.t. displacement
    dh = np.zeros(d)

    # Backprop through levels in reverse order
    # levels[k] = outer(levels[k-1], h).ravel() / (k+1), for k = m-1,...,1 (0-indexed)
    # In the forward, level at 0-index k uses divisor (k+1)
    for k in range(m - 1, 0, -1):
        divisor = k + 1
        scaled = dlevel[k] / divisor
        size_prev = d**k
        mat = scaled.reshape(size_prev, d)

        # Gradient w.r.t. levels[k-1] from the outer product
        dlevel[k - 1] += mat @ h

        # Gradient w.r.t. h from the outer product
        dh += levels[k - 1] @ mat

    # levels[0] = h (identity), so dlevel[0] contributes directly to dh
    dh += dlevel[0]

    return dh


def tensor_log_adjoint(
    dresult: list[NDArray[np.float64]],
    levels: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    """Adjoint of tensor_log: gradient w.r.t. input levels.

    Forward: result = sum_{n=0}^{m-1} (-1)^n / (n+1) * powers[n]
    where powers[0] = x, powers[n] = nil_mul(x, powers[n-1]).

    Backward: propagate through the power chain in reverse.

    Args:
        dresult: Gradient w.r.t. output of tensor_log.
        levels: The input to the forward tensor_log.

    Returns:
        Gradient w.r.t. input levels.
    """
    m = len(levels)

    # Recompute forward powers
    powers = [levels]
    for n in range(1, m):
        powers.append(tensor_multiply_nil(levels, powers[-1]))

    # Direct gradient contribution from each power to the output
    dpowers: list[list[NDArray[np.float64]]] = []
    for n in range(m):
        coeff = (-1) ** n / (n + 1)
        dpowers.append([coeff * dresult[k] for k in range(m)])

    # Backprop through the chain: powers[n] = nil_mul(x, powers[n-1])
    dx = [np.zeros_like(levels[k]) for k in range(m)]

    for n in range(m - 1, 0, -1):
        da, db = _tensor_multiply_nil_adjoint(dpowers[n], levels, powers[n - 1])
        for k in range(m):
            dx[k] += da[k]
            dpowers[n - 1][k] += db[k]

    # dpowers[0] is the accumulated gradient for powers[0] = x
    for k in range(m):
        dx[k] += dpowers[0][k]

    return dx


def _tensor_multiply_nil_adjoint(
    dresult: list[NDArray[np.float64]],
    a: list[NDArray[np.float64]],
    b: list[NDArray[np.float64]],
) -> tuple[list[NDArray[np.float64]], list[NDArray[np.float64]]]:
    """Adjoint of tensor_multiply_nil."""
    m = len(a)
    da = [np.zeros_like(a[k]) for k in range(m)]
    db = [np.zeros_like(b[k]) for k in range(m)]

    if m >= 2:
        buf_da = np.empty(len(a[-1]))
        buf_db = np.empty(len(b[-1]))

    for k in range(m):
        for i in range(k + 1):
            j = k - 1 - i
            if j < 0:
                continue
            size_i = len(a[i])
            size_j = len(b[j])
            dr = dresult[k].reshape(size_i, size_j)
            np.matmul(dr, b[j], out=buf_da[:size_i])
            da[i] += buf_da[:size_i]
            np.matmul(a[i], dr, out=buf_db[:size_j])
            db[j] += buf_db[:size_j]

    return da, db


# --- Batched operations for vectorized computation ---


def tensor_multiply_batch(
    a: list[NDArray[np.float64]],
    b: list[NDArray[np.float64]],
) -> list[NDArray[np.float64]]:
    """Batched tensor multiply (implicit 1 at level 0).

    Both a and b are batched level-lists: each a[k] has shape (batch, d^(k+1)).
    All batch elements are multiplied independently in parallel.

    Args:
        a: Batched level-list of first elements.
        b: Batched level-list of second elements.

    Returns:
        Batched level-list of products.
    """
    m = len(a)
    result = [a[k] + b[k] for k in range(m)]

    for k in range(m):
        for i in range(k + 1):
            j = k - 1 - i
            if j < 0:
                continue
            outer = np.einsum("ni,nj->nij", a[i], b[j])
            result[k] = result[k] + outer.reshape(outer.shape[0], -1)

    return result
