"""Benchmark sig-light against iisignature.

Run with: uv run python scripts/benchmark.py

Requires iisignature to be installed for comparison.
Without it, only sig-light timings are shown.

To install iisignature for cross-validation testing (requires C++ compiler):
    uv pip install setuptools
    uv pip install iisignature --no-build-isolation
"""

import timeit
from collections.abc import Callable
from typing import Any

import numpy as np

import sig_light

try:
    import iisignature  # ty: ignore[unresolved-import]
except ImportError:
    iisignature = None


CONFIGS = [
    ("d=2, m=3, n=50", 2, 3, 50),
    ("d=2, m=5, n=50", 2, 5, 50),
    ("d=3, m=3, n=100", 3, 3, 100),
    ("d=3, m=4, n=100", 3, 4, 100),
    ("d=5, m=3, n=100", 5, 3, 100),
    ("d=3, m=3, n=1000", 3, 3, 1000),
    ("d=2, m=6, n=100", 2, 6, 100),
]

REPEATS = 50


def _time(fn: Callable[[], Any]) -> float:
    """Return average time in seconds over REPEATS runs."""
    return timeit.timeit(fn, number=REPEATS) / REPEATS


def benchmark_sig() -> None:
    """Benchmark sig() computation."""
    rng = np.random.default_rng(42)

    if iisignature is not None:
        print(f"{'Config':<22} {'sig-light':>12} {'iisignature':>12} {'ratio':>8}")
    else:
        print(f"{'Config':<22} {'sig-light':>12}")
    print("-" * (58 if iisignature is not None else 36))

    for label, d, m, n in CONFIGS:
        path = rng.standard_normal((n, d))
        t_ours = _time(lambda: sig_light.sig(path, m))

        if iisignature is not None:
            iisig = iisignature
            t_theirs = _time(lambda: iisig.sig(path, m))
            ratio = t_ours / t_theirs
            print(
                f"{label:<22} {t_ours * 1000:>10.3f}ms"
                f" {t_theirs * 1000:>10.3f}ms {ratio:>7.1f}x"
            )
        else:
            print(f"{label:<22} {t_ours * 1000:>10.3f}ms")


def benchmark_logsig() -> None:
    """Benchmark logsig() computation."""
    rng = np.random.default_rng(42)

    if iisignature is not None:
        print(f"{'Config':<22} {'sig-light':>12} {'iisignature':>12} {'ratio':>8}")
    else:
        print(f"{'Config':<22} {'sig-light':>12}")
    print("-" * (58 if iisignature is not None else 36))

    for label, d, m, n in CONFIGS[:5]:
        path = rng.standard_normal((n, d))
        s_ours = sig_light.prepare(d, m)
        t_ours = _time(lambda: sig_light.logsig(path, s_ours))

        if iisignature is not None:
            iisig = iisignature
            s_theirs = iisig.prepare(d, m)
            t_theirs = _time(lambda: iisig.logsig(path, s_theirs))
            ratio = t_ours / t_theirs
            print(
                f"{label:<22} {t_ours * 1000:>10.3f}ms"
                f" {t_theirs * 1000:>10.3f}ms {ratio:>7.1f}x"
            )
        else:
            print(f"{label:<22} {t_ours * 1000:>10.3f}ms")


if __name__ == "__main__":
    print("=== sig() ===\n")
    benchmark_sig()
    print("\n=== logsig() ===\n")
    benchmark_logsig()
