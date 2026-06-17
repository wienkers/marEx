"""
Pinning the numerical path of xarray's rolling reductions.

xarray 2026.9 stopped routing rolling reductions through bottleneck by default; earlier
releases used it whenever it was installed (it arrives with ``xarray[complete]``). The two
paths round differently in float32, so the same marEx call returned different numbers
depending on the xarray release. Every float rolling mean in marEx runs under
:func:`rolling_numerics`, which selects the numpy path on every release: against a float64
reference it is ~2x more accurate than bottleneck's running sum, and it is independent of
the time chunking, where bottleneck's result moves by up to ~2.7e-4 between two chunkings
(D-141). numbagg is switched off too where xarray has the option: it is another accelerated
path, and it takes numpy-backed windows when installed. The option is read when the graph is
BUILT, so the choice reaches distributed workers.
"""

import xarray as xr
from xarray.core.options import OPTIONS

_PIN = {"use_bottleneck": False, **({"use_numbagg": False} if "use_numbagg" in OPTIONS else {})}


def rolling_numerics():
    """Context manager that pins xarray's rolling reductions to the numpy path (bottleneck and numbagg off)."""
    return xr.set_options(**_PIN)
