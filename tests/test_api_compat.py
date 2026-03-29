"""Cross-validation tests against iisignature.

These tests are skipped if iisignature is not installed.
"""

import numpy as np
import pytest

import sig_light

iisignature = pytest.importorskip("iisignature")


@pytest.fixture
def rng():
    return np.random.default_rng(123)


class TestSigCompat:
    """Compare sig() output against iisignature.sig()."""

    def test_random_2d(self, rng):
        """Random 2D path, depth 4."""
        path = rng.standard_normal((20, 2))
        for m in range(1, 5):
            ours = sig_light.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10, rtol=1e-10)

    def test_random_3d(self, rng):
        """Random 3D path, depth 3."""
        path = rng.standard_normal((15, 3))
        for m in range(1, 4):
            ours = sig_light.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10, rtol=1e-10)

    def test_random_5d(self, rng):
        """Random 5D path, depth 2."""
        path = rng.standard_normal((10, 5))
        for m in [1, 2]:
            ours = sig_light.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10, rtol=1e-10)

    def test_random_1d(self, rng):
        """Random 1D path, depth 5."""
        path = rng.standard_normal((25, 1))
        for m in range(1, 6):
            ours = sig_light.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10, rtol=1e-10)


class TestSiglengthCompat:
    """Compare siglength() against iisignature.siglength()."""

    def test_various(self):
        for d in [1, 2, 3, 5, 10]:
            for m in [1, 2, 3, 4]:
                assert sig_light.siglength(d, m) == iisignature.siglength(d, m)


class TestLogsiglengthCompat:
    """Compare logsiglength() against iisignature.logsiglength()."""

    def test_various(self):
        for d in [1, 2, 3, 5]:
            for m in [1, 2, 3, 4]:
                assert sig_light.logsiglength(d, m) == iisignature.logsiglength(d, m)


class TestLogsigCompat:
    """Compare logsig() output against iisignature.logsig()."""

    def test_random_2d(self, rng):
        """Random 2D path, depth 4."""
        path = rng.standard_normal((20, 2))
        d = 2
        for m in range(1, 5):
            s_ours = sig_light.prepare(d, m)
            s_theirs = iisignature.prepare(d, m)
            ours = sig_light.logsig(path, s_ours)
            theirs = iisignature.logsig(path, s_theirs)
            np.testing.assert_allclose(ours, theirs, atol=1e-10, rtol=1e-10)

    def test_random_3d(self, rng):
        """Random 3D path, depth 3."""
        path = rng.standard_normal((15, 3))
        d = 3
        for m in range(1, 4):
            s_ours = sig_light.prepare(d, m)
            s_theirs = iisignature.prepare(d, m)
            ours = sig_light.logsig(path, s_ours)
            theirs = iisignature.logsig(path, s_theirs)
            np.testing.assert_allclose(ours, theirs, atol=1e-10, rtol=1e-10)


class TestBasisCompat:
    """Compare basis() output against iisignature.basis()."""

    def test_d2(self):
        for m in range(1, 5):
            s_ours = sig_light.prepare(2, m)
            s_theirs = iisignature.prepare(2, m)
            ours = sig_light.basis(s_ours)
            theirs = list(iisignature.basis(s_theirs))
            assert ours == theirs, f"d=2, m={m}: {ours} != {theirs}"

    def test_d3(self):
        for m in range(1, 4):
            s_ours = sig_light.prepare(3, m)
            s_theirs = iisignature.prepare(3, m)
            ours = sig_light.basis(s_ours)
            theirs = list(iisignature.basis(s_theirs))
            assert ours == theirs, f"d=3, m={m}: {ours} != {theirs}"


class TestSigFormat2Compat:
    """Compare sig format=2 against iisignature."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        for m in [2, 3]:
            ours = sig_light.sig(path, m, format=2)
            theirs = iisignature.sig(path, m, 2)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestSigbackpropCompat:
    """Compare sigbackprop against iisignature."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        for m in [2, 3]:
            deriv = rng.standard_normal(iisignature.siglength(2, m))
            ours = sig_light.sigbackprop(deriv, path, m)
            theirs = iisignature.sigbackprop(deriv, path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_3d(self, rng):
        path = rng.standard_normal((8, 3))
        m = 2
        deriv = rng.standard_normal(iisignature.siglength(3, m))
        ours = sig_light.sigbackprop(deriv, path, m)
        theirs = iisignature.sigbackprop(deriv, path, m)
        np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestSigjacobianCompat:
    """Compare sigjacobian against iisignature."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((6, 2))
        for m in [2, 3]:
            ours = sig_light.sigjacobian(path, m)
            theirs = iisignature.sigjacobian(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestLogsigbackpropCompat:
    """Compare logsigbackprop against iisignature."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        for m in [2, 3]:
            s_ours = sig_light.prepare(2, m)
            s_theirs = iisignature.prepare(2, m, "S")
            deriv = rng.standard_normal(iisignature.logsiglength(2, m))
            ours = sig_light.logsigbackprop(deriv, path, s_ours)
            theirs = iisignature.logsigbackprop(deriv, path, s_theirs, "S")
            np.testing.assert_allclose(ours, theirs, atol=1e-8)


class TestSigscaleCompat:
    """Compare sigscale against iisignature."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        scales = rng.standard_normal(2) + 2
        for m in [2, 3]:
            s = sig_light.sig(path, m)
            ours = sig_light.sigscale(s, scales, 2, m)
            theirs = iisignature.sigscale(s, scales, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestBatchingCompat:
    """Compare batched operations against iisignature."""

    def test_batched_sig(self, rng):
        paths = rng.standard_normal((5, 10, 2))
        m = 3
        ours = sig_light.sig(paths, m)
        theirs = iisignature.sig(paths, m)
        np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_batched_logsig(self, rng):
        paths = rng.standard_normal((5, 10, 2))
        m = 3
        s_ours = sig_light.prepare(2, m)
        s_theirs = iisignature.prepare(2, m)
        ours = sig_light.logsig(paths, s_ours)
        theirs = iisignature.logsig(paths, s_theirs)
        np.testing.assert_allclose(ours, theirs, atol=1e-10)
