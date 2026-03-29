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


def _iisig() -> Any:
    """Return iisignature module, typed as Any to satisfy ty in closures."""
    return iisignature


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


def _print_header(has_comparison: bool) -> None:
    if has_comparison:
        print(f"{'Config':<22} {'sig-light':>12} {'iisignature':>12} {'ratio':>8}")
        print("-" * 58)
    else:
        print(f"{'Config':<22} {'sig-light':>12}")
        print("-" * 36)


def _print_row(label: str, t_ours: float, t_theirs: float | None) -> None:
    if t_theirs is not None:
        ratio = t_ours / t_theirs
        print(
            f"{label:<22} {t_ours * 1000:>10.3f}ms"
            f" {t_theirs * 1000:>10.3f}ms {ratio:>7.1f}x"
        )
    else:
        print(f"{label:<22} {t_ours * 1000:>10.3f}ms")


def benchmark_sig() -> None:
    """Benchmark sig() computation."""
    rng = np.random.default_rng(42)
    has = iisignature is not None
    _print_header(has)

    for label, d, m, n in CONFIGS:
        path = rng.standard_normal((n, d))
        t_ours = _time(lambda: sig_light.sig(path, m))
        t_theirs = None
        if has:
            iisig = _iisig()
            t_theirs = _time(lambda: iisig.sig(path, m))
        _print_row(label, t_ours, t_theirs)


def benchmark_logsig() -> None:
    """Benchmark logsig() computation."""
    rng = np.random.default_rng(42)
    has = iisignature is not None
    _print_header(has)

    for label, d, m, n in CONFIGS[:5]:
        path = rng.standard_normal((n, d))
        s_ours = sig_light.prepare(d, m)
        t_ours = _time(lambda: sig_light.logsig(path, s_ours))
        t_theirs = None
        if has:
            iisig = _iisig()
            s_theirs = iisig.prepare(d, m)
            t_theirs = _time(lambda: iisig.logsig(path, s_theirs))
        _print_row(label, t_ours, t_theirs)


def benchmark_sigbackprop() -> None:
    """Benchmark sigbackprop() computation."""
    rng = np.random.default_rng(42)
    has = iisignature is not None
    _print_header(has)

    for label, d, m, n in CONFIGS[:5]:
        path = rng.standard_normal((n, d))
        deriv = rng.standard_normal(sig_light.siglength(d, m))
        t_ours = _time(lambda: sig_light.sigbackprop(deriv, path, m))
        t_theirs = None
        if has:
            iisig = _iisig()
            t_theirs = _time(lambda: iisig.sigbackprop(deriv, path, m))
        _print_row(label, t_ours, t_theirs)


def benchmark_logsigbackprop() -> None:
    """Benchmark logsigbackprop() computation."""
    rng = np.random.default_rng(42)
    has = iisignature is not None
    _print_header(has)

    for label, d, m, n in CONFIGS[:4]:
        path = rng.standard_normal((n, d))
        s_ours = sig_light.prepare(d, m)
        deriv = rng.standard_normal(sig_light.logsiglength(d, m))
        t_ours = _time(lambda: sig_light.logsigbackprop(deriv, path, s_ours))
        t_theirs = None
        if has:
            iisig = _iisig()
            s_theirs = iisig.prepare(d, m, "S")
            t_theirs = _time(lambda: iisig.logsigbackprop(deriv, path, s_theirs, "S"))
        _print_row(label, t_ours, t_theirs)


def benchmark_sigscale() -> None:
    """Benchmark sigscale() computation."""
    rng = np.random.default_rng(42)
    has = iisignature is not None
    _print_header(has)

    for label, d, m, n in CONFIGS[:5]:
        path = rng.standard_normal((n, d))
        s = sig_light.sig(path, m)
        scales = rng.standard_normal(d) + 2
        t_ours = _time(lambda: sig_light.sigscale(s, scales, d, m))
        t_theirs = None
        if has:
            iisig = _iisig()
            t_theirs = _time(lambda: iisig.sigscale(s, scales, m))
        _print_row(label, t_ours, t_theirs)


def benchmark_sigjoin() -> None:
    """Benchmark sigjoin() computation."""
    rng = np.random.default_rng(42)
    has = iisignature is not None
    _print_header(has)

    for label, d, m, n in CONFIGS[:5]:
        path = rng.standard_normal((n, d))
        s = sig_light.sig(path[: n // 2], m)
        seg = path[n // 2] - path[n // 2 - 1]
        t_ours = _time(lambda: sig_light.sigjoin(s, seg, d, m))
        t_theirs = None
        if has:
            iisig = _iisig()
            t_theirs = _time(lambda: iisig.sigjoin(s, seg, m))
        _print_row(label, t_ours, t_theirs)


if __name__ == "__main__":
    print("=== sig() ===\n")
    benchmark_sig()
    print("\n=== logsig() ===\n")
    benchmark_logsig()
    print("\n=== sigbackprop() ===\n")
    benchmark_sigbackprop()
    print("\n=== logsigbackprop() ===\n")
    benchmark_logsigbackprop()
    print("\n=== sigscale() ===\n")
    benchmark_sigscale()
    print("\n=== sigjoin() ===\n")
    benchmark_sigjoin()
