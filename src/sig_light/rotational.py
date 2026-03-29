"""Rotation-invariant features of 2D path signatures.

For d=2 paths, under rotation by angle theta, the signature transforms
via the tensor power of the 2x2 rotation matrix. Rotation-invariant
linear functionals exist only at even signature levels 2, 4, 6, ...

The invariants are found via change-of-basis to complex coordinates
z = x + iy, z* = x - iy, where rotation acts diagonally. Components
with equal numbers of z and z* factors are invariant, giving C(2j, j)
independent real invariants at level 2j.
"""

from dataclasses import dataclass
from itertools import combinations

import numpy as np
from numpy.typing import NDArray

from sig_light.signature import sig


@dataclass(frozen=True)
class RotInv2DPreparedData:
    """Precomputed data for rotation-invariant signature features.

    Created by ``rotinv2dprepare()`` and passed to ``rotinv2d()``.

    Attributes:
        m: Truncation depth of the underlying signature.
        inv_type: Invariant type ("a" for all invariants).
        coefficients: List of coefficient matrices, one per even level.
            Each matrix has shape (n_invariants, 2^level) and extracts
            invariants from the flat signature at that level.
        total_length: Total number of rotation-invariant features.
    """

    m: int
    inv_type: str
    coefficients: list[NDArray[np.float64]]
    total_length: int


def rotinv2dprepare(
    m: int,
    inv_type: str = "a",
) -> RotInv2DPreparedData:
    """Precompute coefficient matrices for rotation-invariant features.

    For 2D paths, rotation invariants exist at even signature levels.
    At level 2j, there are C(2j, j) independent real invariants.

    The invariants are computed by changing basis from (x, y) to
    complex coordinates (z, z*), identifying the invariant subspace
    (balanced multi-indices with equal z and z* counts), and
    converting back to real coefficient vectors.

    Args:
        m: Truncation depth. Invariants are extracted from even levels
            2, 4, ..., 2*(m//2).
        inv_type: Type of invariants. Currently only "a" (all) is
            supported.

    Returns:
        Prepared data for use with rotinv2d().

    Raises:
        ValueError: If inv_type is not supported.
    """
    if inv_type != "a":
        raise ValueError(f"Unsupported inv_type={inv_type!r}. Only 'a' is supported.")

    # Change-of-basis matrix at level 1: columns are z, z* in (x,y) basis
    # z = x + iy  -> [1, i]^T
    # z* = x - iy -> [1, -i]^T
    p1 = np.array([[1, 1], [1j, -1j]], dtype=np.complex128)

    coefficients: list[NDArray[np.float64]] = []
    total = 0

    for j in range(1, m // 2 + 1):
        level = 2 * j
        basis = _invariant_basis_at_level(p1, level, j)
        coefficients.append(basis)
        total += basis.shape[0]

    return RotInv2DPreparedData(
        m=m,
        inv_type=inv_type,
        coefficients=coefficients,
        total_length=total,
    )


def _invariant_basis_at_level(
    p1: NDArray[np.complex128],
    level: int,
    half_level: int,
) -> NDArray[np.float64]:
    """Compute orthonormal invariant coefficient vectors at one even level.

    Args:
        p1: Level-1 change-of-basis matrix, shape (2, 2).
        level: Even signature level (= 2 * half_level).
        half_level: Number of z factors (= number of z* factors).

    Returns:
        Matrix of shape (n_invariants, 2^level) where each row is a
        coefficient vector that extracts one rotation invariant from
        the signature at this level. n_invariants = C(level, half_level).
    """
    # Build P_level = P1^{tensor level}, shape (2^level, 2^level)
    pk = p1
    for _ in range(level - 1):
        pk = np.kron(pk, p1)

    # Find column indices for balanced multi-indices:
    # binary sequences of length `level` with exactly `half_level` ones.
    # In the (z, z*) basis, index 0 = z, index 1 = z*.
    # A binary sequence (b1,...,b_level) maps to column index
    # sum(b_i * 2^(level-1-i)).
    balanced_col_indices = _balanced_indices(level, half_level)

    # Extract complex coefficient vectors (columns of P_k)
    complex_cols = pk[:, balanced_col_indices]

    # Build real candidate vectors from real and imaginary parts
    real_parts = complex_cols.real
    imag_parts = complex_cols.imag
    candidates = np.hstack([real_parts, imag_parts])

    # Find orthonormal basis of the column space via SVD
    u, s, _ = np.linalg.svd(candidates, full_matrices=False)
    tol = max(s[0] * 1e-10, 1e-14) if len(s) > 0 else 1e-14
    rank = int(np.sum(s > tol))

    return u[:, :rank].T.copy()


def _balanced_indices(level: int, half_level: int) -> list[int]:
    """Column indices for balanced binary sequences.

    Returns indices corresponding to binary sequences of length `level`
    with exactly `half_level` ones (z* positions).

    Args:
        level: Total sequence length.
        half_level: Number of 1-bits required.

    Returns:
        Sorted list of integer column indices.
    """
    indices = []
    for ones_positions in combinations(range(level), half_level):
        idx = 0
        for pos in ones_positions:
            idx += 1 << (level - 1 - pos)
        indices.append(idx)
    indices.sort()
    return indices


def rotinv2d(
    path: NDArray[np.float64],
    s: RotInv2DPreparedData,
) -> NDArray[np.float64]:
    """Compute rotation-invariant features of a 2D path signature.

    Extracts linear combinations of signature components that are
    unchanged when the path is rotated by any angle.

    Args:
        path: Array of shape (n, 2) representing a 2D path.
        s: Prepared data from ``rotinv2dprepare()``.

    Returns:
        1D array of length ``rotinv2dlength(s)`` containing the
        rotation-invariant features.

    Raises:
        ValueError: If path is not 2-dimensional.
    """
    path = np.asarray(path, dtype=np.float64)
    if path.ndim != 2 or path.shape[1] != 2:
        raise ValueError(
            f"rotinv2d requires 2D paths with shape (n, 2), got {path.shape}"
        )

    d = 2
    signature = sig(path, s.m)

    result = np.empty(s.total_length)
    sig_offset = 0
    result_offset = 0

    coeff_idx = 0
    for level in range(1, s.m + 1):
        level_size = d**level
        sig_level = signature[sig_offset : sig_offset + level_size]
        sig_offset += level_size

        if level % 2 == 0 and coeff_idx < len(s.coefficients):
            coeffs = s.coefficients[coeff_idx]
            n_inv = coeffs.shape[0]
            result[result_offset : result_offset + n_inv] = coeffs @ sig_level
            result_offset += n_inv
            coeff_idx += 1

    return result


def rotinv2dlength(s: RotInv2DPreparedData) -> int:
    """Number of rotation-invariant features.

    At each even level 2j, there are C(2j, j) invariants.
    Total = sum of C(2j, j) for j = 1, ..., m//2.

    Args:
        s: Prepared data from ``rotinv2dprepare()``.

    Returns:
        Total number of rotation-invariant features.
    """
    return s.total_length


def rotinv2dcoeffs(
    s: RotInv2DPreparedData,
) -> tuple[NDArray[np.float64], ...]:
    """Get the coefficient matrices for each even level.

    Each matrix has shape (n_invariants, 2^level). Multiplying a
    coefficient matrix by the signature at that level gives the
    rotation-invariant features for that level.

    Args:
        s: Prepared data from ``rotinv2dprepare()``.

    Returns:
        Tuple of coefficient matrices, one per even level
        (2, 4, ..., 2*(m//2)).
    """
    return tuple(s.coefficients)
