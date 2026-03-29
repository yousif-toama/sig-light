"""Log signature computation via the signature-then-log (S) method.

The log signature is computed by:
1. Computing the path signature.
2. Taking the tensor logarithm.
3. Projecting onto the Lyndon word basis.
"""

from dataclasses import dataclass

import numpy as np
from numpy.typing import NDArray

from sig_light.algebra import concat_levels, tensor_log
from sig_light.lyndon import (
    build_projection_matrices,
    generate_lyndon_words,
    lyndon_bracket,
)
from sig_light.signature import sig_levels, siglength


@dataclass(frozen=True)
class PreparedData:
    """Precomputed data for log signature computation.

    Created by ``prepare()`` and passed to ``logsig()`` and ``basis()``.
    """

    d: int
    m: int
    lyndon_words: list[tuple[int, ...]]
    basis_labels: list[str]
    projection_matrices: list[NDArray[np.floating]]


def prepare(d: int, m: int) -> PreparedData:
    """Precompute data for log signature computation.

    Generates Lyndon words, their bracket representations, and
    projection matrices for extracting Lyndon coordinates from
    the full tensor logarithm.

    Args:
        d: Path dimension.
        m: Truncation depth.

    Returns:
        Opaque prepared data object for use with logsig() and basis().
    """
    all_words = generate_lyndon_words(d, m)

    # Group by level to match projection output order
    lyndon_words: list[tuple[int, ...]] = []
    basis_labels: list[str] = []
    for k in range(1, m + 1):
        words_at_k = [w for w in all_words if len(w) == k]
        lyndon_words.extend(words_at_k)
        basis_labels.extend(lyndon_bracket(w, one_indexed=True) for w in words_at_k)

    projection_matrices = build_projection_matrices(d, m, all_words)

    return PreparedData(
        d=d,
        m=m,
        lyndon_words=lyndon_words,
        basis_labels=basis_labels,
        projection_matrices=projection_matrices,
    )


def basis(s: PreparedData) -> list[str]:
    """Get the Lyndon bracket labels for the log signature basis.

    Args:
        s: Prepared data from ``prepare()``.

    Returns:
        List of bracket expression strings, one per log signature
        component. E.g. ``["1", "2", "[1,2]"]`` for d=2, m=2.
    """
    return list(s.basis_labels)


def logsig(path: NDArray[np.float64], s: PreparedData) -> NDArray[np.float64]:
    """Compute the log signature of a path.

    Uses the S method: compute signature, take tensor logarithm,
    then project onto the Lyndon basis.

    Args:
        path: Array of shape (..., n, d) representing a piecewise-linear path.
            Extra leading dimensions are batched.
        s: Prepared data from ``prepare(d, m)``.

    Returns:
        Array of shape (..., logsiglength(d, m)) containing the log
        signature in the Lyndon basis.
    """
    path = np.asarray(path, dtype=np.float64)

    if path.ndim > 2:
        return _logsig_batched(path, s)

    return _logsig_single(path, s)


def _logsig_batched(
    path: NDArray[np.float64],
    s: PreparedData,
) -> NDArray[np.float64]:
    """Handle batched paths for logsig."""
    batch_shape = path.shape[:-2]
    n, d = path.shape[-2], path.shape[-1]
    flat_batch = path.reshape(-1, n, d)
    results = np.stack(
        [_logsig_single(flat_batch[i], s) for i in range(flat_batch.shape[0])]
    )
    return results.reshape(*batch_shape, logsiglength(s.d, s.m))


def _logsig_single(
    path: NDArray[np.float64],
    s: PreparedData,
) -> NDArray[np.float64]:
    """Single (non-batched) logsig computation."""
    n = path.shape[0]

    if n < 2:
        return np.zeros(logsiglength(s.d, s.m))

    levels = sig_levels(path, s.m)
    log_levels = tensor_log(levels)

    projected: list[NDArray[np.float64]] = []
    for k in range(s.m):
        proj_matrix = s.projection_matrices[k]
        if proj_matrix.shape[0] == 0:
            continue
        coords = proj_matrix @ log_levels[k]
        projected.append(coords)

    return np.concatenate(projected)


def logsig_expanded(
    path: NDArray[np.float64],
    s: PreparedData,
) -> NDArray[np.float64]:
    """Compute the log signature in the full tensor expansion (X method).

    Returns the tensor logarithm without projecting to the Lyndon basis.
    Output length equals ``siglength(d, m)``.

    Args:
        path: Array of shape (..., n, d). Extra leading dims are batched.
        s: Prepared data from ``prepare(d, m)``.

    Returns:
        Array of shape (..., siglength(d, m)).
    """
    path = np.asarray(path, dtype=np.float64)

    if path.ndim > 2:
        batch_shape = path.shape[:-2]
        n, d = path.shape[-2], path.shape[-1]
        flat_batch = path.reshape(-1, n, d)
        results = np.stack(
            [
                _logsig_expanded_single(flat_batch[i], s)
                for i in range(flat_batch.shape[0])
            ]
        )
        return results.reshape(*batch_shape, siglength(s.d, s.m))

    return _logsig_expanded_single(path, s)


def _logsig_expanded_single(
    path: NDArray[np.float64],
    s: PreparedData,
) -> NDArray[np.float64]:
    """Single (non-batched) logsig_expanded computation."""
    n = path.shape[0]

    if n < 2:
        return np.zeros(siglength(s.d, s.m))

    levels = sig_levels(path, s.m)
    log_levels = tensor_log(levels)
    return concat_levels(log_levels)


def logsiglength(d: int, m: int) -> int:
    """Length of the log signature output (number of Lyndon words up to length m).

    Uses Witt's formula: the number of Lyndon words of length k over d
    letters is ``(1/k) * sum_{j|k} mu(k/j) * d^j``.

    Args:
        d: Path dimension.
        m: Truncation depth.

    Returns:
        Total number of Lyndon words of lengths 1 through m.
    """
    return sum(_necklace_count(d, k) for k in range(1, m + 1))


def _necklace_count(d: int, k: int) -> int:
    """Number of Lyndon words (primitive necklaces) of length k over d letters."""
    total = 0
    for j in _divisors(k):
        total += _mobius(k // j) * d**j
    return total // k


def _divisors(n: int) -> list[int]:
    """All positive divisors of n."""
    divs = []
    for i in range(1, n + 1):
        if n % i == 0:
            divs.append(i)
    return divs


def _mobius(n: int) -> int:
    """Mobius function mu(n).

    Returns:
        1 if n is a product of an even number of distinct primes.
        -1 if n is a product of an odd number of distinct primes.
        0 if n has a squared prime factor.
    """
    if n == 1:
        return 1

    factors = 0
    remaining = n
    d = 2
    while d * d <= remaining:
        if remaining % d == 0:
            remaining //= d
            if remaining % d == 0:
                return 0  # squared factor
            factors += 1
        d += 1 if d == 2 else 2
    if remaining > 1:
        factors += 1

    return 1 if factors % 2 == 0 else -1
