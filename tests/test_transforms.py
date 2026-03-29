"""Tests for signature transforms: join, scale, and their backpropagation."""

import numpy as np

import sig_light


class TestSigjoin:
    """Tests for sigjoin()."""

    def test_extends_signature(self, rng):
        """sigjoin(sig(path[:k], m), path[k]-path[k-1], d, m) == sig(path[:k+1], m)."""
        d, m = 3, 3
        path = rng.standard_normal((10, d))

        for k in range(2, len(path)):
            sig_prefix = sig_light.sig(path[:k], m)
            segment = path[k] - path[k - 1]
            joined = sig_light.sigjoin(sig_prefix, segment, d, m)
            expected = sig_light.sig(path[: k + 1], m)
            np.testing.assert_allclose(joined, expected, atol=1e-10)

    def test_with_fixed_last(self, rng):
        """fixedLast appends a fixed coordinate to the segment."""
        d, m = 3, 2
        path = rng.standard_normal((5, d))
        sig_prefix = sig_light.sig(path[:3], m)

        full_segment = path[3] - path[2]
        partial_segment = full_segment[:-1]
        fixed_val = float(full_segment[-1])

        joined_full = sig_light.sigjoin(sig_prefix, full_segment, d, m)
        joined_fixed = sig_light.sigjoin(
            sig_prefix, partial_segment, d, m, fixedLast=fixed_val
        )
        np.testing.assert_allclose(joined_full, joined_fixed, atol=1e-12)

    def test_1d_path(self, rng):
        """Works for 1D paths."""
        d, m = 1, 3
        path = rng.standard_normal((6, d))

        sig_prefix = sig_light.sig(path[:4], m)
        segment = path[4] - path[3]
        joined = sig_light.sigjoin(sig_prefix, segment, d, m)
        expected = sig_light.sig(path[:5], m)
        np.testing.assert_allclose(joined, expected, atol=1e-10)


class TestSigjoinbackprop:
    """Tests for sigjoinbackprop()."""

    def test_vs_finite_differences(self, rng):
        """Analytical gradients match numerical finite differences."""
        d, m = 2, 3
        sig_flat = sig_light.sig(rng.standard_normal((5, d)), m)
        segment = rng.standard_normal(d)
        deriv = rng.standard_normal(sig_light.siglength(d, m))

        result = sig_light.sigjoinbackprop(deriv, sig_flat, segment, d, m)
        dsig_a, dseg_a = result[0], result[1]

        epsilon = 1e-7

        # Finite diff for dsig
        dsig_num = np.zeros_like(sig_flat)
        for i in range(len(sig_flat)):
            sig_plus = sig_flat.copy()
            sig_minus = sig_flat.copy()
            sig_plus[i] += epsilon
            sig_minus[i] -= epsilon
            f_plus = sig_light.sigjoin(sig_plus, segment, d, m)
            f_minus = sig_light.sigjoin(sig_minus, segment, d, m)
            dsig_num[i] = deriv @ (f_plus - f_minus) / (2 * epsilon)
        np.testing.assert_allclose(dsig_a, dsig_num, atol=1e-5)

        # Finite diff for dsegment
        dseg_num = np.zeros_like(segment)
        for i in range(len(segment)):
            seg_plus = segment.copy()
            seg_minus = segment.copy()
            seg_plus[i] += epsilon
            seg_minus[i] -= epsilon
            f_plus = sig_light.sigjoin(sig_flat, seg_plus, d, m)
            f_minus = sig_light.sigjoin(sig_flat, seg_minus, d, m)
            dseg_num[i] = deriv @ (f_plus - f_minus) / (2 * epsilon)
        np.testing.assert_allclose(dseg_a, dseg_num, atol=1e-5)

    def test_with_fixed_last(self, rng):
        """Backprop through fixedLast returns (dsig, dseg, dfixedLast)."""
        d, m = 3, 2
        sig_flat = sig_light.sig(rng.standard_normal((5, d)), m)
        partial_segment = rng.standard_normal(d - 1)
        fixed_val = 0.5
        deriv = rng.standard_normal(sig_light.siglength(d, m))

        result = sig_light.sigjoinbackprop(
            deriv, sig_flat, partial_segment, d, m, fixedLast=fixed_val
        )
        assert len(result) == 3
        assert result[0].shape == sig_flat.shape
        assert result[1].shape == partial_segment.shape
        assert isinstance(result[-1], float)
        dseg = result[1]

        # Verify dseg via finite differences
        epsilon = 1e-7
        dseg_num = np.zeros_like(partial_segment)
        for i in range(len(partial_segment)):
            seg_plus = partial_segment.copy()
            seg_minus = partial_segment.copy()
            seg_plus[i] += epsilon
            seg_minus[i] -= epsilon
            f_plus = sig_light.sigjoin(sig_flat, seg_plus, d, m, fixedLast=fixed_val)
            f_minus = sig_light.sigjoin(sig_flat, seg_minus, d, m, fixedLast=fixed_val)
            dseg_num[i] = deriv @ (f_plus - f_minus) / (2 * epsilon)
        np.testing.assert_allclose(dseg, dseg_num, atol=1e-5)


class TestSigscale:
    """Tests for sigscale()."""

    def test_identity_with_ones(self, rng):
        """Scaling by ones is the identity."""
        d, m = 3, 3
        path = rng.standard_normal((10, d))
        s = sig_light.sig(path, m)
        ones = np.ones(d)
        scaled = sig_light.sigscale(s, ones, d, m)
        np.testing.assert_allclose(scaled, s, atol=1e-12)

    def test_level_1_scaling(self, rng):
        """Level 1 components are scaled by individual scale factors."""
        d, m = 2, 2
        path = rng.standard_normal((8, d))
        s = sig_light.sig(path, m)
        scales = np.array([2.0, 3.0])

        scaled = sig_light.sigscale(s, scales, d, m)
        # Level 1: each component i scaled by scales[i]
        np.testing.assert_allclose(scaled[:d], s[:d] * scales, atol=1e-12)

    def test_level_k_product_of_scales(self, rng):
        """Level-k entry (i1,...,ik) multiplied by product of scales."""
        d, m = 2, 2
        path = rng.standard_normal((6, d))
        s = sig_light.sig(path, m)
        scales = np.array([2.0, 3.0])

        scaled = sig_light.sigscale(s, scales, d, m)

        # Level 2: entry (i,j) scaled by scales[i]*scales[j]
        sig_level2 = s[d : d + d**2].reshape(d, d)
        scaled_level2 = scaled[d : d + d**2].reshape(d, d)
        for i in range(d):
            for j in range(d):
                expected = sig_level2[i, j] * scales[i] * scales[j]
                np.testing.assert_allclose(scaled_level2[i, j], expected, atol=1e-12)

    def test_d2_m2(self, rng):
        """Specific d=2, m=2 test."""
        d, m = 2, 2
        path = rng.standard_normal((5, d))
        s = sig_light.sig(path, m)
        scales = np.array([0.5, 2.0])

        scaled = sig_light.sigscale(s, scales, d, m)

        # Equivalent to computing sig of the scaled path
        scaled_path = path * scales
        expected = sig_light.sig(scaled_path, m)
        np.testing.assert_allclose(scaled, expected, atol=1e-10)


class TestSigscalebackprop:
    """Tests for sigscalebackprop()."""

    def test_dsig_vs_finite_differences(self, rng):
        """dsig matches numerical finite differences."""
        d, m = 2, 3
        path = rng.standard_normal((6, d))
        sig_flat = sig_light.sig(path, m)
        scales = rng.standard_normal(d) * 0.5 + 1.0
        deriv = rng.standard_normal(sig_light.siglength(d, m))

        dsig_a, _ = sig_light.sigscalebackprop(deriv, sig_flat, scales, d, m)

        epsilon = 1e-7
        dsig_num = np.zeros_like(sig_flat)
        for i in range(len(sig_flat)):
            sig_plus = sig_flat.copy()
            sig_minus = sig_flat.copy()
            sig_plus[i] += epsilon
            sig_minus[i] -= epsilon
            f_plus = sig_light.sigscale(sig_plus, scales, d, m)
            f_minus = sig_light.sigscale(sig_minus, scales, d, m)
            dsig_num[i] = deriv @ (f_plus - f_minus) / (2 * epsilon)
        np.testing.assert_allclose(dsig_a, dsig_num, atol=1e-5)

    def test_dscales_vs_finite_differences(self, rng):
        """dscales matches numerical finite differences."""
        d, m = 2, 2
        path = rng.standard_normal((6, d))
        sig_flat = sig_light.sig(path, m)
        scales = rng.standard_normal(d) * 0.5 + 1.0
        deriv = rng.standard_normal(sig_light.siglength(d, m))

        _, dscales_a = sig_light.sigscalebackprop(deriv, sig_flat, scales, d, m)

        epsilon = 1e-7
        dscales_num = np.zeros(d)
        for i in range(d):
            s_plus = scales.copy()
            s_minus = scales.copy()
            s_plus[i] += epsilon
            s_minus[i] -= epsilon
            f_plus = sig_light.sigscale(sig_flat, s_plus, d, m)
            f_minus = sig_light.sigscale(sig_flat, s_minus, d, m)
            dscales_num[i] = deriv @ (f_plus - f_minus) / (2 * epsilon)
        np.testing.assert_allclose(dscales_a, dscales_num, atol=1e-5)
