"""Signature transforms: join, scale, and their backpropagation adjoints.

These operations extend and rescale flat signatures without recomputing
from the original path.
"""

import math

import numpy as np
from numpy.typing import NDArray

from sig_light.algebra import (
    concat_levels,
    sig_of_segment,
    sig_of_segment_adjoint,
    split_signature,
    tensor_multiply_adjoint,
)
from sig_light.signature import sigcombine


def sigjoin(
    sig_flat: NDArray[np.float64],
    segment: NDArray[np.float64],
    d: int,
    m: int,
    fixedLast: float = float("nan"),
) -> NDArray[np.float64]:
    """Extend a signature by appending a linear segment.

    Computes the signature of the path extended by a single segment
    using Chen's identity: result = sigcombine(sig_flat, sig_of_segment(segment)).

    Args:
        sig_flat: Flat signature of shape (siglength(d, m),).
        segment: Displacement vector of shape (d,), or (d-1,) when
            fixedLast is not NaN.
        d: Path dimension.
        m: Truncation depth.
        fixedLast: If not NaN, appended as the last coordinate of the
            segment displacement.

    Returns:
        Flat signature of shape (siglength(d, m),) for the extended path.
    """
    segment = np.asarray(segment, dtype=np.float64)
    if not math.isnan(fixedLast):
        segment = np.append(segment, fixedLast)
    seg_sig_flat = concat_levels(sig_of_segment(segment, m))
    return sigcombine(sig_flat, seg_sig_flat, d, m)


def sigjoinbackprop(
    deriv: NDArray[np.float64],
    sig_flat: NDArray[np.float64],
    segment: NDArray[np.float64],
    d: int,
    m: int,
    fixedLast: float = float("nan"),
) -> (
    tuple[NDArray[np.float64], NDArray[np.float64]]
    | tuple[NDArray[np.float64], NDArray[np.float64], float]
):
    """Backpropagate through sigjoin.

    Computes gradients of a scalar loss w.r.t. the inputs of sigjoin,
    given the gradient w.r.t. its output.

    Args:
        deriv: Gradient w.r.t. the sigjoin output, shape
            (siglength(d, m),).
        sig_flat: The input signature (forward pass value), shape
            (siglength(d, m),).
        segment: The input displacement (forward pass value), shape
            (d,) or (d-1,) when fixedLast is not NaN.
        d: Path dimension.
        m: Truncation depth.
        fixedLast: Same value used in the forward pass.

    Returns:
        (dsig, dsegment) when fixedLast is NaN, or
        (dsig, dsegment, dfixedLast) when fixedLast is not NaN.
        dsig has shape (siglength(d, m),), dsegment has shape matching
        the input segment, and dfixedLast is a scalar.
    """
    segment = np.asarray(segment, dtype=np.float64)
    used_fixed = not math.isnan(fixedLast)
    full_segment = np.append(segment, fixedLast) if used_fixed else segment

    # Recompute forward values needed for adjoint
    sig_levels = split_signature(np.asarray(sig_flat, dtype=np.float64), d, m)
    seg_levels = sig_of_segment(full_segment, m)

    # Backprop through sigcombine (tensor_multiply)
    deriv_levels = split_signature(np.asarray(deriv, dtype=np.float64), d, m)
    dsig_levels, dseg_levels = tensor_multiply_adjoint(
        deriv_levels, sig_levels, seg_levels
    )

    dsig = concat_levels(dsig_levels)

    # Backprop through sig_of_segment
    dfull_segment = sig_of_segment_adjoint(dseg_levels, full_segment, m)

    if used_fixed:
        dsegment = dfull_segment[:-1]
        dfixedLast = float(dfull_segment[-1])
        return dsig, dsegment, dfixedLast

    return dsig, dfull_segment


def sigscale(
    sig_flat: NDArray[np.float64],
    scales: NDArray[np.float64],
    d: int,
    m: int,
) -> NDArray[np.float64]:
    """Rescale a signature by per-dimension scale factors.

    At level k, the multi-index entry (i1, ..., ik) is multiplied by
    scales[i1] * scales[i2] * ... * scales[ik].

    Args:
        sig_flat: Flat signature of shape (siglength(d, m),).
        scales: Per-dimension scale factors of shape (d,).
        d: Path dimension.
        m: Truncation depth.

    Returns:
        Flat signature of shape (siglength(d, m),) with rescaled entries.
    """
    sig_flat = np.asarray(sig_flat, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)
    levels = split_signature(sig_flat, d, m)

    result_levels: list[NDArray[np.float64]] = []
    for k in range(1, m + 1):
        scale_k = scales.copy()
        for _ in range(k - 1):
            scale_k = np.outer(scale_k, scales).ravel()
        result_levels.append(levels[k - 1] * scale_k)

    return concat_levels(result_levels)


def sigscalebackprop(
    deriv: NDArray[np.float64],
    sig_flat: NDArray[np.float64],
    scales: NDArray[np.float64],
    d: int,
    m: int,
) -> tuple[NDArray[np.float64], NDArray[np.float64]]:
    """Backpropagate through sigscale.

    Computes gradients of a scalar loss w.r.t. the inputs of sigscale,
    given the gradient w.r.t. its output.

    Args:
        deriv: Gradient w.r.t. the sigscale output, shape
            (siglength(d, m),).
        sig_flat: The input signature (forward pass value), shape
            (siglength(d, m),).
        scales: The input scale factors (forward pass value), shape
            (d,).
        d: Path dimension.
        m: Truncation depth.

    Returns:
        (dsig, dscales) where dsig has shape (siglength(d, m),) and
        dscales has shape (d,).
    """
    deriv = np.asarray(deriv, dtype=np.float64)
    sig_flat = np.asarray(sig_flat, dtype=np.float64)
    scales = np.asarray(scales, dtype=np.float64)

    deriv_levels = split_signature(deriv, d, m)
    sig_levels = split_signature(sig_flat, d, m)

    # dsig: sigscale is its own adjoint w.r.t. sig
    dsig = sigscale(deriv, scales, d, m)

    # dscales: accumulate contributions from each level and position
    dscales = np.zeros(d, dtype=np.float64)

    for k in range(1, m + 1):
        # Element-wise product of deriv and sig at this level
        product = deriv_levels[k - 1] * sig_levels[k - 1]

        # Build scale tensor for this level
        scale_k = scales.copy()
        for _ in range(k - 1):
            scale_k = np.outer(scale_k, scales).ravel()

        # For each position pos in the k-fold index (i1, ..., ik),
        # the contribution to dscales[j] is:
        #   product[idx] * (scale_k[idx] / scales[i_pos])
        # summed over all idx where i_pos == j, for all pos.
        shaped = (product * scale_k).reshape([d] * k)

        for pos in range(k):
            # Sum over all axes except the one at position pos,
            # then divide by scales to remove that factor.
            # Sum all axes except pos to get a vector of length d.
            axes = tuple(ax for ax in range(k) if ax != pos)
            contribution = shaped.sum(axis=axes) if axes else shaped.ravel()
            # Divide out the scale factor at position pos.
            # Guard against zero scales with safe division.
            with np.errstate(divide="ignore", invalid="ignore"):
                safe = np.where(scales != 0, contribution / scales, 0.0)
            dscales += safe

    return dsig, dscales
