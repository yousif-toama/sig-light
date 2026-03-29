"""Shared test fixtures for sig-light."""

import numpy as np
import pytest


@pytest.fixture
def l_path():
    """L-shaped 2D path: (0,0) -> (1,0) -> (1,1)."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])


@pytest.fixture
def square_path():
    """Unit square traversal returning to origin."""
    return np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0], [0.0, 0.0]])


@pytest.fixture
def path_1d():
    """1D path: 0 -> 1 -> 3."""
    return np.array([[0.0], [1.0], [3.0]])


@pytest.fixture
def single_point():
    """Degenerate 1-point path."""
    return np.array([[1.0, 2.0]])


@pytest.fixture
def straight_2d():
    """Single straight segment in 2D."""
    return np.array([[0.0, 0.0], [3.0, 5.0]])


@pytest.fixture
def random_3d_path(rng):
    """Random 3D path with 20 points."""
    return rng.standard_normal((20, 3))


@pytest.fixture
def rng():
    """Seeded random number generator for reproducibility."""
    return np.random.default_rng(42)
