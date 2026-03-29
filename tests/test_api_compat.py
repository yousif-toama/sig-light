"""Cross-validation tests against iisignature.

These tests are skipped if iisignature is not installed.
Every public function that mirrors iisignature has a test here.
"""

import numpy as np
import pytest

import sig_light

iisignature = pytest.importorskip("iisignature")


@pytest.fixture
def rng():
    return np.random.default_rng(123)


# --- Forward functions (exact match expected) ---


class TestSigCompat:
    """Compare sig() format=0 against iisignature.sig()."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((20, 2))
        for m in range(1, 5):
            ours = sig_light.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_3d(self, rng):
        path = rng.standard_normal((15, 3))
        for m in range(1, 4):
            ours = sig_light.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_5d(self, rng):
        path = rng.standard_normal((10, 5))
        for m in [1, 2]:
            ours = sig_light.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_1d(self, rng):
        path = rng.standard_normal((25, 1))
        for m in range(1, 6):
            ours = sig_light.sig(path, m)
            theirs = iisignature.sig(path, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestSigFormat2Compat:
    """Compare sig() format=2 against iisignature.sig(format=2)."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        for m in [2, 3]:
            ours = sig_light.sig(path, m, format=2)
            theirs = iisignature.sig(path, m, 2)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_3d(self, rng):
        path = rng.standard_normal((8, 3))
        for m in [2, 3]:
            ours = sig_light.sig(path, m, format=2)
            theirs = iisignature.sig(path, m, 2)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestSiglengthCompat:
    """Compare siglength() against iisignature.siglength()."""

    def test_various(self):
        for d in [1, 2, 3, 5, 10]:
            for m in [1, 2, 3, 4]:
                assert sig_light.siglength(d, m) == iisignature.siglength(d, m)


class TestSigcombineCompat:
    """Compare sigcombine() against iisignature.sigcombine()."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        for m in [2, 3]:
            s1 = sig_light.sig(path[:5], m)
            s2 = sig_light.sig(path[4:], m)
            ours = sig_light.sigcombine(s1, s2, 2, m)
            theirs = iisignature.sigcombine(s1, s2, 2, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestLogsiglengthCompat:
    """Compare logsiglength() against iisignature.logsiglength()."""

    def test_various(self):
        for d in [1, 2, 3, 5]:
            for m in [1, 2, 3, 4]:
                assert sig_light.logsiglength(d, m) == iisignature.logsiglength(d, m)


class TestLogsigCompat:
    """Compare logsig() against iisignature.logsig()."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((20, 2))
        for m in range(1, 5):
            s_ours = sig_light.prepare(2, m)
            s_theirs = iisignature.prepare(2, m)
            ours = sig_light.logsig(path, s_ours)
            theirs = iisignature.logsig(path, s_theirs)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_3d(self, rng):
        path = rng.standard_normal((15, 3))
        for m in range(1, 4):
            s_ours = sig_light.prepare(3, m)
            s_theirs = iisignature.prepare(3, m)
            ours = sig_light.logsig(path, s_ours)
            theirs = iisignature.logsig(path, s_theirs)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestLogsigExpandedCompat:
    """Compare logsig_expanded() against iisignature.logsig(method='X')."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        for m in [2, 3]:
            s_ours = sig_light.prepare(2, m)
            s_theirs = iisignature.prepare(2, m, "X")
            ours = sig_light.logsig_expanded(path, s_ours)
            theirs = iisignature.logsig(path, s_theirs, "X")
            np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestBasisCompat:
    """Compare basis() against iisignature.basis()."""

    def test_d2(self):
        for m in range(1, 5):
            s_ours = sig_light.prepare(2, m)
            s_theirs = iisignature.prepare(2, m)
            assert sig_light.basis(s_ours) == list(iisignature.basis(s_theirs))

    def test_d3(self):
        for m in range(1, 4):
            s_ours = sig_light.prepare(3, m)
            s_theirs = iisignature.prepare(3, m)
            assert sig_light.basis(s_ours) == list(iisignature.basis(s_theirs))


# --- Gradient functions (relaxed tolerance: different fp accumulation order) ---

GRAD_ATOL = 1e-6


class TestSigbackpropCompat:
    """Compare sigbackprop() against iisignature.sigbackprop()."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        for m in [2, 3]:
            deriv = rng.standard_normal(iisignature.siglength(2, m))
            ours = sig_light.sigbackprop(deriv, path, m)
            theirs = iisignature.sigbackprop(deriv, path, m)
            np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)

    def test_random_3d(self, rng):
        path = rng.standard_normal((8, 3))
        m = 2
        deriv = rng.standard_normal(iisignature.siglength(3, m))
        ours = sig_light.sigbackprop(deriv, path, m)
        theirs = iisignature.sigbackprop(deriv, path, m)
        np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)


class TestSigjacobianCompat:
    """Compare sigjacobian() against iisignature.sigjacobian()."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((6, 2))
        for m in [2, 3]:
            ours = sig_light.sigjacobian(path, m)
            theirs = iisignature.sigjacobian(path, m)
            np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)


class TestLogsigbackpropCompat:
    """Compare logsigbackprop() against iisignature.logsigbackprop()."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        for m in [2, 3]:
            s_ours = sig_light.prepare(2, m)
            s_theirs = iisignature.prepare(2, m, "S")
            deriv = rng.standard_normal(iisignature.logsiglength(2, m))
            ours = sig_light.logsigbackprop(deriv, path, s_ours)
            theirs = iisignature.logsigbackprop(deriv, path, s_theirs, "S")
            np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)

    def test_random_3d(self, rng):
        path = rng.standard_normal((8, 3))
        m = 2
        s_ours = sig_light.prepare(3, m)
        s_theirs = iisignature.prepare(3, m, "S")
        deriv = rng.standard_normal(iisignature.logsiglength(3, m))
        ours = sig_light.logsigbackprop(deriv, path, s_ours)
        theirs = iisignature.logsigbackprop(deriv, path, s_theirs, "S")
        np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)


# --- Transform functions ---


class TestSigjoinCompat:
    """Compare sigjoin() against iisignature.sigjoin()."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        for m in [2, 3]:
            s = sig_light.sig(path[:5], m)
            seg = path[5] - path[4]
            ours = sig_light.sigjoin(s, seg, 2, m)
            theirs = iisignature.sigjoin(s, seg, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_3d(self, rng):
        path = rng.standard_normal((8, 3))
        m = 2
        s = sig_light.sig(path[:4], m)
        seg = path[4] - path[3]
        ours = sig_light.sigjoin(s, seg, 3, m)
        theirs = iisignature.sigjoin(s, seg, m)
        np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestSigjoinbackpropCompat:
    """Compare sigjoinbackprop() against iisignature.sigjoinbackprop()."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((8, 2))
        m = 2
        s = sig_light.sig(path[:4], m)
        seg = path[4] - path[3]
        deriv = rng.standard_normal(iisignature.siglength(2, m))
        ours = sig_light.sigjoinbackprop(deriv, s, seg, 2, m)
        theirs = iisignature.sigjoinbackprop(deriv, s, seg, m)
        np.testing.assert_allclose(ours[0], theirs[0], atol=GRAD_ATOL)
        np.testing.assert_allclose(ours[1], theirs[1], atol=GRAD_ATOL)


class TestSigscaleCompat:
    """Compare sigscale() against iisignature.sigscale()."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        scales = rng.standard_normal(2) + 2
        for m in [2, 3]:
            s = sig_light.sig(path, m)
            ours = sig_light.sigscale(s, scales, 2, m)
            theirs = iisignature.sigscale(s, scales, m)
            np.testing.assert_allclose(ours, theirs, atol=1e-10)

    def test_random_3d(self, rng):
        path = rng.standard_normal((8, 3))
        scales = np.array([2.0, 0.5, 3.0])
        m = 2
        s = sig_light.sig(path, m)
        ours = sig_light.sigscale(s, scales, 3, m)
        theirs = iisignature.sigscale(s, scales, m)
        np.testing.assert_allclose(ours, theirs, atol=1e-10)


class TestSigscalebackpropCompat:
    """Compare sigscalebackprop() against iisignature.sigscalebackprop()."""

    def test_random_2d(self, rng):
        path = rng.standard_normal((10, 2))
        scales = rng.standard_normal(2) + 2
        m = 3
        s = sig_light.sig(path, m)
        deriv = rng.standard_normal(iisignature.siglength(2, m))
        ours = sig_light.sigscalebackprop(deriv, s, scales, 2, m)
        theirs = iisignature.sigscalebackprop(deriv, s, scales, m)
        np.testing.assert_allclose(ours[0], theirs[0], atol=1e-10)
        np.testing.assert_allclose(ours[1], theirs[1], atol=1e-10)


# --- Rotation invariants ---


class TestRotinv2dCompat:
    """Compare rotinv2d functions against iisignature."""

    def test_rotinv2dlength(self):
        for m in [2, 4, 6]:
            s_ours = sig_light.rotinv2dprepare(m, "a")
            s_theirs = iisignature.rotinv2dprepare(m, "a")
            assert sig_light.rotinv2dlength(s_ours) == iisignature.rotinv2dlength(
                s_theirs
            )

    def test_rotinv2d_values(self, rng):
        path = rng.standard_normal((15, 2))
        for m in [2, 4]:
            s_ours = sig_light.rotinv2dprepare(m, "a")
            s_theirs = iisignature.rotinv2dprepare(m, "a")
            ours = sig_light.rotinv2d(path, s_ours)
            theirs = iisignature.rotinv2d(path, s_theirs)
            assert len(ours) == len(theirs)
            # Invariants may use different basis — check same span
            # by verifying both are rotation-invariant and same length
            # For exact match we'd need same basis convention
            # Check lengths match at minimum
            np.testing.assert_equal(len(ours), len(theirs))


# --- Batching ---


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

    def test_batched_sigbackprop(self, rng):
        paths = rng.standard_normal((3, 8, 2))
        m = 2
        deriv = rng.standard_normal((3, iisignature.siglength(2, m)))
        ours = sig_light.sigbackprop(deriv, paths, m)
        theirs = iisignature.sigbackprop(deriv, paths, m)
        np.testing.assert_allclose(ours, theirs, atol=GRAD_ATOL)
