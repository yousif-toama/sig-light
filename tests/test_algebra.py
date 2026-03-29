"""Tests for truncated tensor algebra and Lyndon word operations."""

import numpy as np
import pytest

from sig_light.algebra import (
    concat_levels,
    sig_of_segment,
    split_signature,
    tensor_log,
    tensor_multiply,
    tensor_multiply_nil,
)
from sig_light.lyndon import (
    _is_lyndon,
    build_projection_matrices,
    generate_lyndon_words,
    standard_factorization,
)


class TestTensorMultiply:
    """Tests for tensor_multiply (implicit 1 at level 0)."""

    def test_identity_right(self):
        """Multiplying by zero element (implicit 1) is identity."""
        d, m = 2, 3
        a = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0, 6.0]), np.zeros(8)]
        zero = [np.zeros(d**k) for k in range(1, m + 1)]

        result = tensor_multiply(a, zero)
        for k in range(m):
            np.testing.assert_allclose(result[k], a[k])

    def test_identity_left(self):
        """Zero element on the left is identity."""
        d, m = 2, 3
        b = [np.array([1.0, 2.0]), np.array([3.0, 4.0, 5.0, 6.0]), np.zeros(8)]
        zero = [np.zeros(d**k) for k in range(1, m + 1)]

        result = tensor_multiply(zero, b)
        for k in range(m):
            np.testing.assert_allclose(result[k], b[k])

    def test_associativity(self, rng):
        """(a * b) * c == a * (b * c)."""
        d, m = 2, 2
        a = [rng.standard_normal(d**k) for k in range(1, m + 1)]
        b = [rng.standard_normal(d**k) for k in range(1, m + 1)]
        c = [rng.standard_normal(d**k) for k in range(1, m + 1)]

        ab_c = tensor_multiply(tensor_multiply(a, b), c)
        a_bc = tensor_multiply(a, tensor_multiply(b, c))

        for k in range(m):
            np.testing.assert_allclose(ab_c[k], a_bc[k], atol=1e-12)

    def test_level_1_additive(self):
        """Level 1 of the product is the sum of level 1 components."""
        a = [np.array([1.0, 2.0]), np.zeros(4)]
        b = [np.array([3.0, 4.0]), np.zeros(4)]

        result = tensor_multiply(a, b)
        np.testing.assert_allclose(result[0], [4.0, 6.0])


class TestTensorMultiplyNil:
    """Tests for tensor_multiply_nil (implicit 0 at level 0)."""

    def test_simple_outer_product(self):
        """Level 2 of nil product is outer product of level 1 inputs."""
        a = [np.array([1.0, 2.0]), np.zeros(4)]
        b = [np.array([3.0, 4.0]), np.zeros(4)]

        result = tensor_multiply_nil(a, b)
        np.testing.assert_allclose(result[0], np.zeros(2))
        expected = np.outer([1.0, 2.0], [3.0, 4.0]).ravel()
        np.testing.assert_allclose(result[1], expected)

    def test_zero_at_level_1(self):
        """Level 1 of nil product is always zero (no identity contribution)."""
        a = [np.array([5.0, 6.0]), np.ones(4)]
        b = [np.array([7.0, 8.0]), np.ones(4)]

        result = tensor_multiply_nil(a, b)
        np.testing.assert_allclose(result[0], np.zeros(2))


class TestTensorLog:
    """Tests for tensor logarithm."""

    def test_log_of_zero_is_zero(self):
        """log(1 + 0) = 0."""
        m, d = 3, 2
        zero = [np.zeros(d**k) for k in range(1, m + 1)]
        result = tensor_log(zero)
        for k in range(m):
            np.testing.assert_allclose(result[k], np.zeros_like(result[k]))

    def test_log_of_single_segment(self):
        """For a linear segment, log(exp(h)) should recover h."""
        h = np.array([2.0, 3.0])
        m = 4
        levels = sig_of_segment(h, m)
        log_levels = tensor_log(levels)

        # Level 1 should be h
        np.testing.assert_allclose(log_levels[0], h, atol=1e-12)
        # Higher levels should be ~0
        for k in range(1, m):
            np.testing.assert_allclose(
                log_levels[k], np.zeros_like(log_levels[k]), atol=1e-12
            )

    def test_log_preserves_antisymmetric_part(self):
        """The level-2 antisymmetric part encodes signed area."""
        # Two segments that enclose area
        h1 = np.array([1.0, 0.0])
        h2 = np.array([0.0, 1.0])
        m = 2

        s1 = sig_of_segment(h1, m)
        s2 = sig_of_segment(h2, m)
        combined = tensor_multiply(s1, s2)
        log_combined = tensor_log(combined)

        # Level 1: net displacement
        np.testing.assert_allclose(log_combined[0], [1.0, 1.0], atol=1e-12)
        # Level 2: antisymmetric part = [0, 0.5, -0.5, 0]
        expected = np.array([0.0, 0.5, -0.5, 0.0])
        np.testing.assert_allclose(log_combined[1], expected, atol=1e-12)


class TestSigOfSegment:
    """Tests for truncated exponential of a linear segment."""

    def test_level_1(self):
        """Level 1 is the displacement itself."""
        h = np.array([2.0, 3.0, -1.0])
        levels = sig_of_segment(h, 3)
        np.testing.assert_allclose(levels[0], h)

    def test_level_2(self):
        """Level 2 is h tensor h / 2."""
        h = np.array([1.0, 2.0])
        levels = sig_of_segment(h, 2)
        expected = np.outer(h, h).ravel() / 2
        np.testing.assert_allclose(levels[1], expected)

    def test_level_3(self):
        """Level 3 is h tensor h tensor h / 6."""
        h = np.array([1.0, 2.0])
        levels = sig_of_segment(h, 3)
        expected = np.einsum("i,j,k->ijk", h, h, h).ravel() / 6
        np.testing.assert_allclose(levels[2], expected)

    def test_depth_1(self):
        """Depth-1 signature is just the displacement."""
        h = np.array([5.0, -3.0])
        levels = sig_of_segment(h, 1)
        assert len(levels) == 1
        np.testing.assert_allclose(levels[0], h)


class TestSplitConcat:
    """Tests for split_signature and concat_levels roundtrip."""

    def test_roundtrip(self, rng):
        """split then concat recovers the original flat array."""
        d, m = 3, 3
        length = sum(d**k for k in range(1, m + 1))
        flat = rng.standard_normal(length)

        levels = split_signature(flat, d, m)
        recovered = concat_levels(levels)
        np.testing.assert_allclose(recovered, flat)

    def test_level_sizes(self):
        """Each level has the correct size d^k."""
        d, m = 2, 4
        length = sum(d**k for k in range(1, m + 1))
        flat = np.zeros(length)

        levels = split_signature(flat, d, m)
        assert len(levels) == m
        for k in range(m):
            assert levels[k].shape == (d ** (k + 1),)


class TestIsLyndon:
    """Tests for _is_lyndon helper."""

    def test_empty_word(self):
        assert _is_lyndon(()) is False

    def test_single_letter(self):
        assert _is_lyndon((0,)) is True
        assert _is_lyndon((1,)) is True

    def test_lyndon_word(self):
        assert _is_lyndon((0, 1)) is True

    def test_non_lyndon(self):
        assert _is_lyndon((1, 0)) is False
        assert _is_lyndon((0, 0)) is False


class TestStandardFactorization:
    """Tests for standard_factorization."""

    def test_length_2(self):
        u, v = standard_factorization((0, 1))
        assert u == (0,)
        assert v == (1,)

    def test_length_3(self):
        u, v = standard_factorization((0, 0, 1))
        assert u == (0,)
        assert v == (0, 1)

    def test_invalid_raises(self):
        """Non-Lyndon words with no valid factorization raise ValueError."""
        with pytest.raises(ValueError, match="No standard factorization"):
            standard_factorization((0,))


class TestBuildProjectionMatrices:
    """Tests for build_projection_matrices edge cases."""

    def test_empty_level(self):
        """Levels with no Lyndon words produce empty matrices."""
        # d=1 has no Lyndon words of length > 1
        words = generate_lyndon_words(1, 3)
        matrices = build_projection_matrices(1, 3, words)
        assert matrices[0].shape == (1, 1)  # level 1: just (0,)
        assert matrices[1].shape == (0, 1)  # level 2: no words
        assert matrices[2].shape == (0, 1)  # level 3: no words
