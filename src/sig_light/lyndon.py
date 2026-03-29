"""Lyndon word generation, bracketing, and basis projection.

Lyndon words form a basis for the free Lie algebra, which is used to
represent log signatures compactly. This module provides:

- Generation of all Lyndon words up to a given length (Duval's algorithm).
- Standard (Chen-Fox-Lyndon) factorization of Lyndon words.
- Conversion of bracketed Lyndon words to tensor representations.
- Construction of projection matrices for extracting Lyndon coordinates
  from the full tensor log.
"""

import numpy as np
from numpy.typing import NDArray


def generate_lyndon_words(
    d: int,
    max_length: int,
) -> list[tuple[int, ...]]:
    """Generate all Lyndon words over alphabet {0, ..., d-1} up to max_length.

    Uses a variant of Duval's algorithm. Lyndon words are primitive
    necklaces: strings strictly smaller than all their proper rotations.

    Args:
        d: Alphabet size (number of path dimensions).
        max_length: Maximum word length to generate.

    Returns:
        List of Lyndon words as tuples, grouped by length then
        lexicographic order. Each letter is in {0, ..., d-1}.
    """
    words: list[tuple[int, ...]] = []
    _duval_generate(d, max_length, words)
    return words


def _duval_generate(
    d: int,
    max_length: int,
    result: list[tuple[int, ...]],
) -> None:
    """Generate Lyndon words using Duval's algorithm."""
    w = [-1]
    while w:
        # Increment last character
        w[-1] += 1
        if len(w) <= max_length:
            # Output if this is a complete Lyndon word
            result.append(tuple(w))
        # Repeat pattern to fill up to max_length
        i = 0
        while len(w) < max_length:
            w.append(w[i])
            i += 1
        # Find rightmost character that can be incremented
        while w and w[-1] == d - 1:
            w.pop()


def standard_factorization(
    word: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    """Find the standard (CFL) factorization of a Lyndon word.

    The standard factorization of a Lyndon word w is the unique
    split w = uv where v is the longest proper Lyndon suffix of w.

    Args:
        word: A Lyndon word as a tuple of integers.

    Returns:
        Pair (u, v) where w = u + v, both u and v are Lyndon,
        and u < v lexicographically.
    """
    n = len(word)
    # Find longest proper Lyndon suffix by checking all suffixes
    for split in range(1, n):
        suffix = word[split:]
        if _is_lyndon(suffix):
            return word[:split], suffix
    msg = f"No standard factorization found for {word}"
    raise ValueError(msg)


def _is_lyndon(word: tuple[int, ...]) -> bool:
    """Check whether a word is a Lyndon word."""
    n = len(word)
    if n == 0:
        return False
    for i in range(1, n):
        rotation = word[i:] + word[:i]
        if rotation <= word:
            return False
    return True


def lyndon_bracket(
    word: tuple[int, ...],
    one_indexed: bool = True,
) -> str:
    """Convert a Lyndon word to its standard Lie bracket expression.

    Single letters are displayed as their index. Multi-letter words
    are recursively bracketed using the standard factorization.

    Args:
        word: A Lyndon word as a tuple of integers (0-indexed).
        one_indexed: If True, display letters as 1-indexed (matching
            iisignature convention).

    Returns:
        String bracket expression, e.g. "[1,2]" or "[[1,2],2]".
    """
    if len(word) == 1:
        idx = word[0] + 1 if one_indexed else word[0]
        return str(idx)
    u, v = standard_factorization(word)
    return f"[{lyndon_bracket(u, one_indexed)},{lyndon_bracket(v, one_indexed)}]"


def lyndon_to_tensor(
    word: tuple[int, ...],
    d: int,
) -> NDArray[np.float64]:
    """Expand a Lyndon word to its tensor representation via Lie brackets.

    Single letter i maps to the standard basis vector e_i in R^d.
    For a composite word with standard factorization w = uv,
    the tensor is [T(u), T(v)] = T(u) tensor T(v) - T(v) tensor T(u).

    Args:
        word: A Lyndon word (0-indexed letters in {0, ..., d-1}).
        d: Alphabet size / path dimension.

    Returns:
        1D array of shape (d^len(word),) representing the tensor.
    """
    if len(word) == 1:
        e = np.zeros(d)
        e[word[0]] = 1.0
        return e

    u, v = standard_factorization(word)
    t_u = lyndon_to_tensor(u, d)
    t_v = lyndon_to_tensor(v, d)

    # Lie bracket: [u, v] = u tensor v - v tensor u
    return np.outer(t_u, t_v).ravel() - np.outer(t_v, t_u).ravel()


def build_projection_matrices(
    d: int,
    m: int,
    lyndon_words: list[tuple[int, ...]],
) -> list[NDArray[np.floating]]:
    """Build projection matrices for extracting Lyndon coordinates.

    For each level k, constructs a matrix whose columns are the tensor
    representations of Lyndon words of length k, then computes its
    pseudoinverse for projecting from full tensor space to Lyndon
    coordinates.

    Args:
        d: Path dimension.
        m: Maximum depth.
        lyndon_words: All Lyndon words up to length m.

    Returns:
        List of m matrices. For level k (1-indexed), the matrix at
        index k-1 has shape (num_lyndon_words_of_length_k, d^k) such
        that ``matrix @ tensor_level_k`` gives Lyndon coordinates.
    """
    matrices: list[NDArray[np.floating]] = []

    for k in range(1, m + 1):
        words_at_k = [w for w in lyndon_words if len(w) == k]

        if not words_at_k:
            matrices.append(np.zeros((0, d**k)))
            continue

        # Build matrix: each column is a Lyndon tensor
        basis_matrix = np.column_stack([lyndon_to_tensor(w, d) for w in words_at_k])

        # Pseudoinverse for projection: coords = pinv(basis) @ tensor
        pinv = np.linalg.pinv(basis_matrix)
        matrices.append(pinv)

    return matrices
