"""Tests for log signature computation."""

import numpy as np

import sig_light
from sig_light.logsignature import logsiglength


class TestLogsiglength:
    """Tests for logsiglength()."""

    def test_d2_m2(self):
        assert logsiglength(2, 2) == 3

    def test_d2_m3(self):
        assert logsiglength(2, 3) == 5

    def test_d3_m3(self):
        assert logsiglength(3, 3) == 14

    def test_level_1_equals_d(self):
        """At depth 1, logsiglength == d (just the letters)."""
        for d in range(1, 6):
            assert logsiglength(d, 1) == d

    def test_d1_all_depths(self):
        """For d=1, logsiglength(1, m) = 1 for all m >= 1."""
        for m in range(1, 8):
            assert logsiglength(1, m) == 1

    def test_matches_basis_count(self):
        """logsiglength matches the actual number of basis elements."""
        for d in range(2, 5):
            for m in range(1, 5):
                s = sig_light.prepare(d, m)
                assert len(sig_light.basis(s)) == logsiglength(d, m)


class TestPrepareAndBasis:
    """Tests for prepare() and basis()."""

    def test_basis_d2_m2(self):
        s = sig_light.prepare(2, 2)
        b = sig_light.basis(s)
        assert b == ["1", "2", "[1,2]"]

    def test_basis_d2_m3(self):
        s = sig_light.prepare(2, 3)
        b = sig_light.basis(s)
        assert b == ["1", "2", "[1,2]", "[1,[1,2]]", "[[1,2],2]"]

    def test_basis_d3_m1(self):
        s = sig_light.prepare(3, 1)
        assert sig_light.basis(s) == ["1", "2", "3"]

    def test_basis_length_matches_logsiglength(self):
        """Basis length matches logsiglength for various (d, m)."""
        for d in [2, 3, 4]:
            for m in [1, 2, 3, 4]:
                s = sig_light.prepare(d, m)
                assert len(sig_light.basis(s)) == logsiglength(d, m)

    def test_prepare_is_reusable(self, rng):
        """Same prepared data can be used for multiple paths."""
        d, m = 2, 3
        s = sig_light.prepare(d, m)

        path1 = rng.standard_normal((10, d))
        path2 = rng.standard_normal((15, d))

        ls1 = sig_light.logsig(path1, s)
        ls2 = sig_light.logsig(path2, s)

        assert ls1.shape == (logsiglength(d, m),)
        assert ls2.shape == (logsiglength(d, m),)
        assert not np.allclose(ls1, ls2)


class TestLogsig:
    """Tests for logsig()."""

    def test_single_point_is_zero(self, single_point):
        """1-point path gives zero log signature."""
        s = sig_light.prepare(2, 3)
        ls = sig_light.logsig(single_point, s)
        np.testing.assert_allclose(ls, 0.0)

    def test_straight_line_level1_only(self, straight_2d):
        """Straight line: log sig has only level-1 components."""
        s = sig_light.prepare(2, 3)
        ls = sig_light.logsig(straight_2d, s)
        b = sig_light.basis(s)

        # Level-1 components (single letters)
        level1_idx = [i for i, label in enumerate(b) if len(label) == 1]
        level_higher_idx = [i for i, label in enumerate(b) if len(label) > 1]

        displacement = straight_2d[-1] - straight_2d[0]
        np.testing.assert_allclose(ls[level1_idx], displacement, atol=1e-12)
        np.testing.assert_allclose(ls[level_higher_idx], 0.0, atol=1e-12)

    def test_square_area(self, square_path):
        """Square path: zero displacement, area = 1 in [1,2] component."""
        s = sig_light.prepare(2, 2)
        ls = sig_light.logsig(square_path, s)
        b = sig_light.basis(s)

        # Should be [0, 0, 1] for basis ['1', '2', '[1,2]']
        np.testing.assert_allclose(ls[b.index("1")], 0.0, atol=1e-12)
        np.testing.assert_allclose(ls[b.index("2")], 0.0, atol=1e-12)
        np.testing.assert_allclose(ls[b.index("[1,2]")], 1.0, atol=1e-12)

    def test_output_length(self, rng):
        """Output length matches logsiglength."""
        d, m = 3, 3
        path = rng.standard_normal((10, d))
        s = sig_light.prepare(d, m)
        ls = sig_light.logsig(path, s)
        assert ls.shape == (logsiglength(d, m),)

    def test_l_path(self, l_path):
        """L-path log signature: displacement [1,1], area 0.5."""
        s = sig_light.prepare(2, 2)
        ls = sig_light.logsig(l_path, s)
        expected = np.array([1.0, 1.0, 0.5])
        np.testing.assert_allclose(ls, expected, atol=1e-12)

    def test_1d_path(self):
        """1D log signature: only level 1 has a Lyndon word."""
        path = np.array([[0.0], [1.0], [3.0]])
        s = sig_light.prepare(1, 3)
        ls = sig_light.logsig(path, s)
        # For d=1, log sig is just the net displacement
        np.testing.assert_allclose(ls, [3.0], atol=1e-12)


class TestLogsigExpanded:
    """Tests for logsig_expanded (X method)."""

    def test_straight_line(self, straight_2d):
        """Expanded log sig of straight line: level 1 = displacement."""
        s = sig_light.prepare(2, 2)
        ls = sig_light.logsig_expanded(straight_2d, s)

        displacement = straight_2d[-1] - straight_2d[0]
        assert ls.shape == (sig_light.siglength(2, 2),)
        np.testing.assert_allclose(ls[:2], displacement, atol=1e-12)
        np.testing.assert_allclose(ls[2:], 0.0, atol=1e-12)

    def test_output_length(self, rng):
        """Expanded log sig length matches siglength."""
        d, m = 3, 3
        path = rng.standard_normal((10, d))
        s = sig_light.prepare(d, m)
        ls = sig_light.logsig_expanded(path, s)
        assert ls.shape == (sig_light.siglength(d, m),)

    def test_single_point_is_zero(self, single_point):
        """1-point path gives zero expanded log signature."""
        s = sig_light.prepare(2, 2)
        ls = sig_light.logsig_expanded(single_point, s)
        assert ls.shape == (sig_light.siglength(2, 2),)
        np.testing.assert_allclose(ls, 0.0)
