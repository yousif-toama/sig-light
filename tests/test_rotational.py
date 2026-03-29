"""Tests for rotation-invariant features of 2D path signatures."""

import math

import numpy as np
import pytest

import sig_light


class TestRotinv2dprepare:
    """Tests for rotinv2dprepare() and rotinv2dlength()."""

    def test_m2_length(self):
        """m=2: C(2,1)=2 invariants."""
        s = sig_light.rotinv2dprepare(2)
        assert sig_light.rotinv2dlength(s) == 2

    def test_m4_length(self):
        """m=4: C(2,1) + C(4,2) = 2 + 6 = 8 invariants."""
        s = sig_light.rotinv2dprepare(4)
        assert sig_light.rotinv2dlength(s) == 8

    def test_m6_length(self):
        """m=6: C(2,1) + C(4,2) + C(6,3) = 2 + 6 + 20 = 28 invariants."""
        s = sig_light.rotinv2dprepare(6)
        assert sig_light.rotinv2dlength(s) == 28

    def test_odd_m_same_as_even_below(self):
        """Odd m uses even levels up to m, so m=3 same as m=2."""
        s3 = sig_light.rotinv2dprepare(3)
        s2 = sig_light.rotinv2dprepare(2)
        assert sig_light.rotinv2dlength(s3) == sig_light.rotinv2dlength(s2)

    def test_unsupported_type_raises(self):
        """Unsupported inv_type raises ValueError."""
        with pytest.raises(ValueError, match="Unsupported inv_type"):
            sig_light.rotinv2dprepare(2, inv_type="b")


class TestRotinv2d:
    """Tests for rotinv2d()."""

    def test_rotation_invariance(self, rng):
        """Rotating the path by a random angle gives the same features."""
        m = 4
        s = sig_light.rotinv2dprepare(m)
        path = rng.standard_normal((15, 2))
        inv_original = sig_light.rotinv2d(path, s)

        for _ in range(3):
            theta = rng.uniform(0, 2 * math.pi)
            c, sn = math.cos(theta), math.sin(theta)
            rot = np.array([[c, -sn], [sn, c]])
            rotated_path = path @ rot.T
            inv_rotated = sig_light.rotinv2d(rotated_path, s)
            np.testing.assert_allclose(inv_original, inv_rotated, atol=1e-10)

    def test_simple_known_path(self):
        """L-shaped path produces nonzero invariants."""
        m = 2
        s = sig_light.rotinv2dprepare(m)
        path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
        inv = sig_light.rotinv2d(path, s)
        assert inv.shape == (sig_light.rotinv2dlength(s),)
        assert not np.allclose(inv, 0.0)

    def test_various_m(self, rng):
        """Output length matches expected for m=2, 4, 6."""
        path = rng.standard_normal((10, 2))
        for m in [2, 4, 6]:
            s = sig_light.rotinv2dprepare(m)
            inv = sig_light.rotinv2d(path, s)
            assert inv.shape == (sig_light.rotinv2dlength(s),)

    def test_non_2d_raises(self):
        """3D paths raise ValueError."""
        s = sig_light.rotinv2dprepare(2)
        path_3d = np.array([[0.0, 0.0, 0.0], [1.0, 1.0, 1.0]])
        with pytest.raises(ValueError, match="2D paths"):
            sig_light.rotinv2d(path_3d, s)


class TestRotinv2dcoeffs:
    """Tests for rotinv2dcoeffs()."""

    def test_returns_tuple(self):
        """Returns a tuple of arrays."""
        s = sig_light.rotinv2dprepare(4)
        coeffs = sig_light.rotinv2dcoeffs(s)
        assert isinstance(coeffs, tuple)

    def test_correct_number_of_levels(self):
        """Number of coefficient matrices equals m//2."""
        for m in [2, 4, 6]:
            s = sig_light.rotinv2dprepare(m)
            coeffs = sig_light.rotinv2dcoeffs(s)
            assert len(coeffs) == m // 2

    def test_correct_shapes(self):
        """Each coefficient matrix has shape (n_inv, 2^level)."""
        s = sig_light.rotinv2dprepare(4)
        coeffs = sig_light.rotinv2dcoeffs(s)

        # Level 2: C(2,1) = 2 invariants, 2^2 = 4 sig components
        assert coeffs[0].shape == (2, 4)
        # Level 4: C(4,2) = 6 invariants, 2^4 = 16 sig components
        assert coeffs[1].shape == (6, 16)
