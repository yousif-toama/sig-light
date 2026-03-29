"""Backward pass (reverse-mode differentiation) for signature and log signature.

Provides gradient computation for:
- sigbackprop: gradient of a scalar loss through the signature w.r.t. path.
- sigjacobian: full Jacobian of the signature w.r.t. path.
- logsigbackprop: gradient of a scalar loss through the log signature w.r.t. path.
"""

from __future__ import annotations

import numpy as np
from numpy.typing import NDArray

from sig_light.algebra import (
    sig_of_segment_adjoint_batch,
    sig_of_segment_batch,
    split_signature,
    tensor_log_adjoint,
    tensor_multiply,
    tensor_multiply_adjoint,
)
from sig_light.logsignature import PreparedData
from sig_light.signature import sig_levels, siglength


def sigbackprop(
    deriv: NDArray[np.float64],
    path: NDArray[np.float64],
    m: int,
) -> NDArray[np.float64]:
    """Compute the gradient of a scalar loss through the signature w.r.t. the path.

    Given dL/dS (the gradient of loss w.r.t. the flat signature), computes
    dL/dpath by backpropagating through the Chen's identity fold.

    Args:
        deriv: Gradient w.r.t. signature, shape (..., siglength(d, m)).
        path: Array of shape (..., n, d).
        m: Truncation depth.

    Returns:
        Gradient w.r.t. path, shape (..., n, d).
    """
    path = np.asarray(path, dtype=np.float64)
    deriv = np.asarray(deriv, dtype=np.float64)

    if path.ndim > 2:
        batch_shape = path.shape[:-2]
        n, d = path.shape[-2], path.shape[-1]
        flat_path = path.reshape(-1, n, d)
        flat_deriv = deriv.reshape(-1, deriv.shape[-1])
        results = np.stack(
            [
                sigbackprop(flat_deriv[i], flat_path[i], m)
                for i in range(flat_path.shape[0])
            ]
        )
        return results.reshape(*batch_shape, n, d)

    n, d = path.shape

    if n < 2:
        return np.zeros_like(path)

    displacements = np.diff(path, axis=0)  # (n-1, d)
    return _sigbackprop_core(deriv, displacements, d, m)


def _sigbackprop_core(
    deriv: NDArray[np.float64],
    displacements: NDArray[np.float64],
    d: int,
    m: int,
) -> NDArray[np.float64]:
    """Core sigbackprop logic operating on displacements.

    Args:
        deriv: Flat gradient of shape (siglength(d, m),).
        displacements: Array of shape (num_segs, d).
        d: Path dimension.
        m: Truncation depth.

    Returns:
        Gradient w.r.t. path, shape (num_segs + 1, d).
    """
    num_segs = displacements.shape[0]

    # Batch compute all segment signatures at once
    seg_batch = sig_of_segment_batch(displacements, m)
    # Convert to list-of-level-lists for the fold
    seg_sigs = [[seg_batch[k][i] for k in range(m)] for i in range(num_segs)]

    # Sequential forward fold (must store intermediates for backprop)
    acc = [seg_sigs[0]]
    for i in range(1, num_segs):
        acc.append(tensor_multiply(acc[-1], seg_sigs[i]))

    # Backward through the fold
    dlevels = split_signature(deriv, d, m)
    dseg = _backprop_fold(dlevels, acc, seg_sigs)

    # Batched: convert segment gradients to displacement gradients
    # Stack dseg into batched level-lists
    dseg_batch = [np.stack([dseg[i][k] for i in range(num_segs)]) for k in range(m)]
    dh = sig_of_segment_adjoint_batch(dseg_batch, displacements, m)

    # Vectorized: convert displacement gradients to path gradients
    # dpath[0] = -dh[0], dpath[i] = dh[i-1] - dh[i], dpath[-1] = dh[-1]
    n = num_segs + 1
    dpath = np.zeros((n, d))
    dpath[:-1] -= dh
    dpath[1:] += dh

    return dpath


def _backprop_fold(
    dacc_final: list[NDArray[np.float64]],
    acc: list[list[NDArray[np.float64]]],
    seg_sigs: list[list[NDArray[np.float64]]],
) -> list[list[NDArray[np.float64]]]:
    """Backpropagate through the left fold of tensor multiplications.

    The forward fold is:
        acc[0] = seg[0]
        acc[i] = tensor_multiply(acc[i-1], seg[i])  for i >= 1

    Args:
        dacc_final: Gradient w.r.t. acc[num_segs - 1] (the final signature).
        acc: Forward accumulator values.
        seg_sigs: Segment signatures.

    Returns:
        List of gradients w.r.t. each segment signature.
    """
    num_segs = len(seg_sigs)
    m = len(seg_sigs[0])

    dseg: list[list[NDArray[np.float64]]] = [
        [np.zeros_like(seg_sigs[i][k]) for k in range(m)] for i in range(num_segs)
    ]

    dacc_i = dacc_final
    for i in range(num_segs - 1, 0, -1):
        da, db = tensor_multiply_adjoint(dacc_i, acc[i - 1], seg_sigs[i])
        dacc_i = da
        for k in range(m):
            dseg[i][k] += db[k]

    # dseg[0] = dacc[0] since acc[0] = seg[0]
    for k in range(m):
        dseg[0][k] += dacc_i[k]

    return dseg


def sigjacobian(
    path: NDArray[np.float64],
    m: int,
) -> NDArray[np.float64]:
    """Compute the full Jacobian of the signature w.r.t. the path.

    jacobian[a, b, c] is the derivative of signature component c
    w.r.t. path[a, b].

    Args:
        path: Array of shape (n, d).
        m: Truncation depth.

    Returns:
        Array of shape (n, d, siglength(d, m)).
    """
    path = np.asarray(path, dtype=np.float64)
    n, d = path.shape
    sig_len = siglength(d, m)

    jacobian = np.zeros((n, d, sig_len))

    e = np.zeros(sig_len)
    for c in range(sig_len):
        e[c] = 1.0
        jacobian[:, :, c] = sigbackprop(e, path, m)
        e[c] = 0.0

    return jacobian


def logsigbackprop(
    deriv: NDArray[np.float64],
    path: NDArray[np.float64],
    s: PreparedData,
) -> NDArray[np.float64]:
    """Compute the gradient of a scalar loss through the log signature w.r.t. path.

    Backpropagates through: path -> signature -> tensor_log -> projection.

    Args:
        deriv: Gradient w.r.t. log signature, shape (..., logsiglength(d, m)).
        path: Array of shape (..., n, d).
        s: Prepared data from ``prepare()``.

    Returns:
        Gradient w.r.t. path, shape (..., n, d).
    """
    path = np.asarray(path, dtype=np.float64)
    deriv = np.asarray(deriv, dtype=np.float64)

    if path.ndim > 2:
        batch_shape = path.shape[:-2]
        n, d = path.shape[-2], path.shape[-1]
        flat_path = path.reshape(-1, n, d)
        flat_deriv = deriv.reshape(-1, deriv.shape[-1])
        results = np.stack(
            [
                logsigbackprop(flat_deriv[i], flat_path[i], s)
                for i in range(flat_path.shape[0])
            ]
        )
        return results.reshape(*batch_shape, n, d)

    n, d = path.shape
    m = s.m

    if n < 2:
        return np.zeros_like(path)

    # Step 1: Unproject from Lyndon basis to full tensor levels
    deriv = np.asarray(deriv, dtype=np.float64)
    dlog_levels = _unproject_lyndon(deriv, s)

    # Step 2: Backprop through tensor_log
    sig_levs = sig_levels(path, m)
    dsig_levels = tensor_log_adjoint(dlog_levels, sig_levs)

    # Step 3: Backprop through sig computation (reuse core logic)
    displacements = np.diff(path, axis=0)
    num_segs = n - 1

    # Batch compute segment signatures
    seg_batch = sig_of_segment_batch(displacements, m)
    seg_sigs = [[seg_batch[k][i] for k in range(m)] for i in range(num_segs)]

    acc = [seg_sigs[0]]
    for i in range(1, num_segs):
        acc.append(tensor_multiply(acc[-1], seg_sigs[i]))

    dseg = _backprop_fold(dsig_levels, acc, seg_sigs)

    # Batched adjoint
    dseg_batch = [np.stack([dseg[i][k] for i in range(num_segs)]) for k in range(m)]
    dh = sig_of_segment_adjoint_batch(dseg_batch, displacements, m)

    # Vectorized path gradient
    dpath = np.zeros_like(path)
    dpath[:-1] -= dh
    dpath[1:] += dh

    return dpath


def _unproject_lyndon(
    deriv: NDArray[np.float64],
    s: PreparedData,
) -> list[NDArray[np.float64]]:
    """Backpropagate through Lyndon basis projection to get tensor-level gradients.

    Forward: coords[k] = projection_matrices[k] @ log_levels[k]
    Backward: dlog_levels[k] = projection_matrices[k].T @ dcoords[k]

    Args:
        deriv: Flat gradient w.r.t. log signature.
        s: Prepared data containing projection matrices.

    Returns:
        Level-list of gradients w.r.t. tensor log levels.
    """
    dlog_levels: list[NDArray[np.float64]] = []
    offset = 0

    for k in range(s.m):
        proj_matrix = s.projection_matrices[k]
        num_coords = proj_matrix.shape[0]

        if num_coords == 0:
            dlog_levels.append(np.zeros(s.d ** (k + 1)))
            continue

        dcoords = deriv[offset : offset + num_coords]
        dlog_levels.append(proj_matrix.T @ dcoords)
        offset += num_coords

    return dlog_levels
