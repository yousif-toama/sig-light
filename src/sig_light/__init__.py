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

from sig_light.backprop import logsigbackprop, sigbackprop, sigjacobian
from sig_light.logsignature import (
    PreparedData,
    basis,
    logsig,
    logsig_expanded,
    logsiglength,
    prepare,
)
from sig_light.rotational import (
    RotInv2DPreparedData,
    rotinv2d,
    rotinv2dcoeffs,
    rotinv2dlength,
    rotinv2dprepare,
)
from sig_light.signature import sig, sigcombine, siglength
from sig_light.transforms import (
    sigjoin,
    sigjoinbackprop,
    sigscale,
    sigscalebackprop,
)

__version__ = "0.2.5"

__all__ = [
    "PreparedData",
    "RotInv2DPreparedData",
    "__version__",
    "basis",
    "logsig",
    "logsig_expanded",
    "logsigbackprop",
    "logsiglength",
    "prepare",
    "rotinv2d",
    "rotinv2dcoeffs",
    "rotinv2dlength",
    "rotinv2dprepare",
    "sig",
    "sigbackprop",
    "sigcombine",
    "sigjacobian",
    "sigjoin",
    "sigjoinbackprop",
    "siglength",
    "sigscale",
    "sigscalebackprop",
    "version",
]


def version() -> str:
    """Return the sig-light version string."""
    return __version__
