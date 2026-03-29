"""sig-light: Pure-Python path signature and log signature computation.

A lightweight, pure-Python library for computing path signatures and log
signatures of multidimensional time series, mirroring the iisignature API.

Example::

    import numpy as np
    import sig_light

    path = np.array([[0.0, 0.0], [1.0, 0.0], [1.0, 1.0]])
    s = sig_light.sig(path, 2)
    print(s)  # [1. 1. 0.5 1. 0. 0.5]
"""

from sig_light.logsignature import (
    PreparedData,
    basis,
    logsig,
    logsig_expanded,
    logsiglength,
    prepare,
)
from sig_light.signature import sig, sigcombine, siglength

__version__ = "0.1.0"

__all__ = [
    "PreparedData",
    "__version__",
    "basis",
    "logsig",
    "logsig_expanded",
    "logsiglength",
    "prepare",
    "sig",
    "sigcombine",
    "siglength",
    "version",
]


def version() -> str:
    """Return the sig-light version string."""
    return __version__
