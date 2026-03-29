"""Tests for backward pass (reverse-mode differentiation) of signatures."""

import numpy as np

import sig_light


def _numerical_sig_gradient(path, m, deriv, epsilon=1e-7):
    """Compute dL/dpath numerically via finite differences.

    L = dot(deriv, sig(path, m))
    """
    path = np.asarray(path, dtype=np.float64)
    grad = np.zeros_like(path)
    for i in range(path.shape[0]):
        for j in range(path.shape[1]):
            path_plus = path.copy()
            path_minus = path.copy()
            path_plus[i, j] += epsilon
            path_minus[i, j] -= epsilon
            s_plus = sig_light.sig(path_plus, m)
            s_minus = sig_light.sig(path_minus, m)
            grad[i, j] = deriv @ (s_plus - s_minus) / (2 * epsilon)
    return grad


def _numerical_logsig_gradient(path, s, deriv, epsilon=1e-7):
    """Compute dL/dpath numerically for logsig via finite differences."""
    path = np.asarray(path, dtype=np.float64)
    grad = np.zeros_like(path)
    for i in range(path.shape[0]):
        for j in range(path.shape[1]):
            path_plus = path.copy()
            path_minus = path.copy()
            path_plus[i, j] += epsilon
            path_minus[i, j] -= epsilon
            ls_plus = sig_light.logsig(path_plus, s)
            ls_minus = sig_light.logsig(path_minus, s)
            grad[i, j] = deriv @ (ls_plus - ls_minus) / (2 * epsilon)
    return grad


class TestSigbackprop:
    """Tests for sigbackprop()."""

    def test_vs_finite_differences(self, rng):
        """Analytical gradient matches numerical finite differences."""
        d, m = 2, 3
        path = rng.standard_normal((8, d))
        deriv = rng.standard_normal(sig_light.siglength(d, m))

        analytical = sig_light.sigbackprop(deriv, path, m)
        numerical = _numerical_sig_gradient(path, m, deriv)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    def test_vs_sigjacobian(self, rng):
        """sigbackprop(d, path, m) == einsum('c,abc->ab', d, sigjacobian)."""
        d, m = 2, 2
        path = rng.standard_normal((6, d))
        deriv = rng.standard_normal(sig_light.siglength(d, m))

        bp = sig_light.sigbackprop(deriv, path, m)
        jac = sig_light.sigjacobian(path, m)
        via_jac = np.einsum("c,abc->ab", deriv, jac)
        np.testing.assert_allclose(bp, via_jac, atol=1e-12)

    def test_two_point_path(self, rng):
        """Gradient through a 2-point (single segment) path."""
        d, m = 3, 2
        path = rng.standard_normal((2, d))
        deriv = rng.standard_normal(sig_light.siglength(d, m))

        analytical = sig_light.sigbackprop(deriv, path, m)
        numerical = _numerical_sig_gradient(path, m, deriv)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    def test_various_d_and_m(self, rng):
        """Gradient is correct for different dimensions and depths."""
        for d, m in [(1, 3), (3, 2), (2, 4)]:
            path = rng.standard_normal((5, d))
            deriv = rng.standard_normal(sig_light.siglength(d, m))

            analytical = sig_light.sigbackprop(deriv, path, m)
            numerical = _numerical_sig_gradient(path, m, deriv)
            np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    def test_single_point_returns_zeros(self, single_point):
        """A single-point path has zero gradient."""
        d = single_point.shape[1]
        m = 3
        deriv = np.ones(sig_light.siglength(d, m))
        grad = sig_light.sigbackprop(deriv, single_point, m)
        np.testing.assert_allclose(grad, 0.0)
        assert grad.shape == single_point.shape


class TestSigjacobian:
    """Tests for sigjacobian()."""

    def test_shape(self, rng):
        """Jacobian has shape (n, d, siglength)."""
        d, m = 2, 3
        path = rng.standard_normal((7, d))
        jac = sig_light.sigjacobian(path, m)
        assert jac.shape == (7, d, sig_light.siglength(d, m))

    def test_consistency_with_sigbackprop(self, rng):
        """Each column of the Jacobian matches a sigbackprop call."""
        d, m = 2, 2
        path = rng.standard_normal((5, d))
        jac = sig_light.sigjacobian(path, m)
        sig_len = sig_light.siglength(d, m)

        for c in range(sig_len):
            e = np.zeros(sig_len)
            e[c] = 1.0
            grad = sig_light.sigbackprop(e, path, m)
            np.testing.assert_allclose(jac[:, :, c], grad, atol=1e-12)

    def test_vs_finite_differences(self, rng):
        """Full Jacobian matches numerical finite differences."""
        d, m = 2, 2
        path = rng.standard_normal((5, d))
        jac = sig_light.sigjacobian(path, m)
        epsilon = 1e-7

        numerical_jac = np.zeros_like(jac)
        for a in range(path.shape[0]):
            for b in range(d):
                path_plus = path.copy()
                path_minus = path.copy()
                path_plus[a, b] += epsilon
                path_minus[a, b] -= epsilon
                s_plus = sig_light.sig(path_plus, m)
                s_minus = sig_light.sig(path_minus, m)
                numerical_jac[a, b, :] = (s_plus - s_minus) / (2 * epsilon)

        np.testing.assert_allclose(jac, numerical_jac, atol=1e-5)


class TestLogsigbackprop:
    """Tests for logsigbackprop()."""

    def test_vs_finite_differences(self, rng):
        """Analytical gradient matches numerical finite differences."""
        d, m = 2, 3
        path = rng.standard_normal((8, d))
        s = sig_light.prepare(d, m)
        deriv = rng.standard_normal(sig_light.logsiglength(d, m))

        analytical = sig_light.logsigbackprop(deriv, path, s)
        numerical = _numerical_logsig_gradient(path, s, deriv)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    def test_two_point_path(self, rng):
        """Two-point path: logsig = displacement, gradient is simple."""
        d, m = 2, 3
        path = rng.standard_normal((2, d))
        s = sig_light.prepare(d, m)
        deriv = rng.standard_normal(sig_light.logsiglength(d, m))

        analytical = sig_light.logsigbackprop(deriv, path, s)
        numerical = _numerical_logsig_gradient(path, s, deriv)
        np.testing.assert_allclose(analytical, numerical, atol=1e-5)

    def test_single_point_returns_zeros(self, single_point):
        """A single-point path has zero gradient."""
        d = single_point.shape[1]
        m = 3
        s = sig_light.prepare(d, m)
        deriv = np.ones(sig_light.logsiglength(d, m))
        grad = sig_light.logsigbackprop(deriv, single_point, s)
        np.testing.assert_allclose(grad, 0.0)
        assert grad.shape == single_point.shape

    def test_1d_path(self, rng):
        """1D path with depth > 1 hits empty Lyndon level branch."""
        path = rng.standard_normal((5, 1))
        s = sig_light.prepare(1, 3)
        deriv = np.ones(sig_light.logsiglength(1, 3))
        grad = sig_light.logsigbackprop(deriv, path, s)
        assert grad.shape == (5, 1)

    def test_batched(self, rng):
        """Batched logsigbackprop matches individual."""
        d, m = 2, 2
        paths = rng.standard_normal((3, 6, d))
        s = sig_light.prepare(d, m)
        derivs = rng.standard_normal((3, sig_light.logsiglength(d, m)))
        result = sig_light.logsigbackprop(derivs, paths, s)
        assert result.shape == (3, 6, d)
        for i in range(3):
            individual = sig_light.logsigbackprop(derivs[i], paths[i], s)
            np.testing.assert_allclose(result[i], individual, atol=1e-12)
