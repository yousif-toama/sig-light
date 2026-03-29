"""Tests for path signature computation."""

import numpy as np

import sig_light


class TestVersion:
    """Tests for version()."""

    def test_version_returns_string(self):
        assert isinstance(sig_light.version(), str)

    def test_version_matches_dunder(self):
        assert sig_light.version() == sig_light.__version__


class TestSiglength:
    """Tests for siglength()."""

    def test_d2_m2(self):
        assert sig_light.siglength(2, 2) == 6

    def test_d3_m3(self):
        assert sig_light.siglength(3, 3) == 3 + 9 + 27

    def test_d1(self):
        """For d=1, siglength(1, m) = m."""
        assert sig_light.siglength(1, 5) == 5

    def test_formula(self):
        """Check against the explicit formula d*(d^m - 1) / (d - 1)."""
        for d in range(2, 6):
            for m in range(1, 5):
                expected = d * (d**m - 1) // (d - 1)
                assert sig_light.siglength(d, m) == expected

    def test_large(self):
        """Large values should work without overflow."""
        assert sig_light.siglength(9500, 2) == 9500**2 + 9500


class TestSig:
    """Tests for sig()."""

    def test_l_path_depth2(self, l_path):
        """L-shaped path: known analytical signature."""
        s = sig_light.sig(l_path, 2)
        expected = np.array([1.0, 1.0, 0.5, 1.0, 0.0, 0.5])
        np.testing.assert_allclose(s, expected, atol=1e-12)

    def test_square_path_depth2(self, square_path):
        """Square path: net displacement zero, area terms nonzero."""
        s = sig_light.sig(square_path, 2)
        expected = np.array([0.0, 0.0, 0.0, 1.0, -1.0, 0.0])
        np.testing.assert_allclose(s, expected, atol=1e-12)

    def test_1d_depth3(self, path_1d):
        """1D path [0] -> [1] -> [3], depth 3."""
        s = sig_light.sig(path_1d, 3)
        expected = np.array([3.0, 4.5, 4.5])
        np.testing.assert_allclose(s, expected, atol=1e-12)

    def test_single_point_is_zero(self, single_point):
        """1-point path gives zero signature."""
        s = sig_light.sig(single_point, 3)
        assert s.shape == (sig_light.siglength(2, 3),)
        np.testing.assert_allclose(s, 0.0)

    def test_level_1_is_displacement(self, rng):
        """Level 1 of signature equals net displacement."""
        path = rng.standard_normal((15, 4))
        s = sig_light.sig(path, 3)
        displacement = path[-1] - path[0]
        np.testing.assert_allclose(s[:4], displacement, atol=1e-12)

    def test_output_length(self, rng):
        """Output length matches siglength."""
        d, m = 3, 4
        path = rng.standard_normal((10, d))
        s = sig_light.sig(path, m)
        assert s.shape == (sig_light.siglength(d, m),)

    def test_format_1_returns_list(self, l_path):
        """format=1 returns a list of arrays, one per level."""
        levels = sig_light.sig(l_path, 3, format=1)
        assert isinstance(levels, list)
        assert len(levels) == 3
        assert levels[0].shape == (2,)
        assert levels[1].shape == (4,)
        assert levels[2].shape == (8,)

    def test_format_1_matches_format_0(self, rng):
        """format=1 concatenated equals format=0."""
        path = rng.standard_normal((10, 3))
        flat = sig_light.sig(path, 3, format=0)
        levels = sig_light.sig(path, 3, format=1)
        np.testing.assert_allclose(np.concatenate(levels), flat)

    def test_straight_line_depth3(self, straight_2d):
        """Straight line signature: level k = h^{tensor k} / k!."""
        h = np.array([3.0, 5.0])
        s = sig_light.sig(straight_2d, 3, format=1)

        np.testing.assert_allclose(s[0], h, atol=1e-12)
        np.testing.assert_allclose(s[1], np.outer(h, h).ravel() / 2, atol=1e-12)
        expected_3 = np.einsum("i,j,k->ijk", h, h, h).ravel() / 6
        np.testing.assert_allclose(s[2], expected_3, atol=1e-12)

    def test_two_point_path(self):
        """Two-point path is a single segment."""
        path = np.array([[1.0, 2.0], [4.0, -1.0]])
        s = sig_light.sig(path, 2)
        h = np.array([3.0, -3.0])
        expected = np.concatenate([h, np.outer(h, h).ravel() / 2])
        np.testing.assert_allclose(s, expected, atol=1e-12)


class TestSigcombine:
    """Tests for sigcombine (Chen's identity)."""

    def test_chens_identity(self, rng):
        """sig(full_path) == sigcombine(sig(first_half), sig(second_half))."""
        d, m = 3, 3
        path = rng.standard_normal((20, d))

        mid = 10
        sig_full = sig_light.sig(path, m)
        sig_first = sig_light.sig(path[: mid + 1], m)
        sig_second = sig_light.sig(path[mid:], m)

        combined = sig_light.sigcombine(sig_first, sig_second, d, m)
        np.testing.assert_allclose(combined, sig_full, atol=1e-10)

    def test_chens_identity_multiple_splits(self, rng):
        """Chen's identity holds for multiple split points."""
        d, m = 2, 3
        path = rng.standard_normal((15, d))

        for split in [3, 7, 12]:
            sig_full = sig_light.sig(path, m)
            sig_a = sig_light.sig(path[: split + 1], m)
            sig_b = sig_light.sig(path[split:], m)
            combined = sig_light.sigcombine(sig_a, sig_b, d, m)
            np.testing.assert_allclose(combined, sig_full, atol=1e-10)

    def test_identity_element(self, rng):
        """Combining with zero signature (identity) returns original."""
        d, m = 2, 3
        path = rng.standard_normal((10, d))
        s = sig_light.sig(path, m)
        zero = np.zeros(sig_light.siglength(d, m))

        np.testing.assert_allclose(sig_light.sigcombine(s, zero, d, m), s, atol=1e-12)
        np.testing.assert_allclose(sig_light.sigcombine(zero, s, d, m), s, atol=1e-12)


class TestSigFormat2:
    """Tests for sig() with format=2 (cumulative prefix signatures)."""

    def test_output_shape(self, rng):
        """format=2 returns shape (n-1, siglength)."""
        d, m = 3, 3
        path = rng.standard_normal((10, d))
        result = sig_light.sig(path, m, format=2)
        assert result.shape == (9, sig_light.siglength(d, m))

    def test_last_row_matches_full_sig(self, rng):
        """Last row matches sig(path, m)."""
        d, m = 2, 3
        path = rng.standard_normal((8, d))
        cumulative = sig_light.sig(path, m, format=2)
        full = sig_light.sig(path, m)
        np.testing.assert_allclose(cumulative[-1], full, atol=1e-12)

    def test_each_row_matches_prefix(self, rng):
        """Row i matches sig(path[:i+2], m)."""
        d, m = 2, 3
        path = rng.standard_normal((7, d))
        cumulative = sig_light.sig(path, m, format=2)

        for i in range(len(path) - 1):
            expected = sig_light.sig(path[: i + 2], m)
            np.testing.assert_allclose(cumulative[i], expected, atol=1e-12)

    def test_single_point_returns_empty(self, single_point):
        """Single-point path returns shape (0, siglength)."""
        d = single_point.shape[1]
        m = 3
        result = sig_light.sig(single_point, m, format=2)
        assert result.shape == (0, sig_light.siglength(d, m))

    def test_batched_format2(self, rng):
        """Batched format=2 matches individual computation."""
        d, m = 2, 2
        batch = rng.standard_normal((3, 6, d))
        result = sig_light.sig(batch, m, format=2)
        assert result.shape == (3, 5, sig_light.siglength(d, m))

        for i in range(3):
            individual = sig_light.sig(batch[i], m, format=2)
            np.testing.assert_allclose(result[i], individual, atol=1e-12)


class TestSigBatching:
    """Tests for batched sig() computation."""

    def test_batch_shape(self, rng):
        """Batch of paths: shape (B, n, d) -> (B, siglength)."""
        d, m = 2, 3
        batch = rng.standard_normal((4, 10, d))
        result = sig_light.sig(batch, m)
        assert result.shape == (4, sig_light.siglength(d, m))

    def test_batch_matches_individual(self, rng):
        """Each batch element matches individual computation."""
        d, m = 2, 3
        batch = rng.standard_normal((5, 8, d))
        result = sig_light.sig(batch, m)

        for i in range(5):
            individual = sig_light.sig(batch[i], m)
            np.testing.assert_allclose(result[i], individual, atol=1e-12)

    def test_multidim_batch(self, rng):
        """Multi-dimensional batch: (B1, B2, n, d) -> (B1, B2, siglength)."""
        d, m = 2, 2
        batch = rng.standard_normal((3, 4, 6, d))
        result = sig_light.sig(batch, m)
        assert result.shape == (3, 4, sig_light.siglength(d, m))

        for i in range(3):
            for j in range(4):
                individual = sig_light.sig(batch[i, j], m)
                np.testing.assert_allclose(result[i, j], individual, atol=1e-12)


class TestSigcombineBatching:
    """Tests for batched sigcombine()."""

    def test_batch_matches_individual(self, rng):
        """Batched sigcombine matches individual computation."""
        d, m = 2, 3
        batch_size = 5
        sl = sig_light.siglength(d, m)

        sig1_batch = rng.standard_normal((batch_size, sl))
        sig2_batch = rng.standard_normal((batch_size, sl))

        result = sig_light.sigcombine(sig1_batch, sig2_batch, d, m)
        assert result.shape == (batch_size, sl)

        for i in range(batch_size):
            individual = sig_light.sigcombine(sig1_batch[i], sig2_batch[i], d, m)
            np.testing.assert_allclose(result[i], individual, atol=1e-12)
