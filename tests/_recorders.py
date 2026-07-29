"""Instruments shared by the area test packages."""

from __future__ import annotations

import numbers
import sys

import dask


class PersistRecorder:
    """Record the size of every object materialised by ``dask.persist``, including through module-level bindings.

    Patching ``dask.persist`` alone misses any marEx module that did ``from dask import persist``:
    that name was bound at import time and still points at the original function. The recorder
    therefore also rebinds every ``persist`` attribute of an imported ``marEx.*`` module that *is*
    ``dask.persist``, and on exit restores every ``marEx.*`` module bound to the recorder,
    including one first imported inside the block (it bound the patched name). ``DataArray.persist``
    and ``Dataset.persist`` need no patch of their own: xarray reaches dask through
    ``from dask import persist`` at call time, which reads the patched attribute.

    A recorder that saw nothing is reported as an error on exit, so a recorder that is not wired
    cannot make a byte bound pass.

    Attributes
    ----------
    calls : list of tuple of int
        One entry per ``persist`` call: the ``nbytes`` of each object handed to it.
    """

    def __init__(self, allow_empty=False):
        self.allow_empty = allow_empty
        self.calls = []
        self._real = None
        self._bound = None

    @property
    def bytes(self):
        """Total bytes handed to ``persist`` across every call."""
        return sum(sum(sizes) for sizes in self.calls)

    @property
    def largest(self):
        """The largest single object handed to ``persist``."""
        return max((size for sizes in self.calls for size in sizes), default=0)

    def _persist(self, *args, **kwargs):
        sizes = []
        for obj in args:
            nbytes = getattr(obj, "nbytes", 0)
            sizes.append(int(nbytes) if isinstance(nbytes, numbers.Integral) else 0)
        self.calls.append(tuple(sizes))
        return self._real(*args, **kwargs)

    @staticmethod
    def _marex_modules():
        return [module for name, module in list(sys.modules.items()) if name == "marEx" or name.startswith("marEx.")]

    def __enter__(self):
        self._real = real = dask.persist
        # One bound-method object, so the exit scan can find it by identity.
        self._bound = self._persist
        dask.persist = self._bound
        for module in self._marex_modules():
            if getattr(module, "persist", None) is real:
                module.persist = self._bound
        return self

    def __exit__(self, exc_type, exc, tb):
        dask.persist = self._real
        for module in self._marex_modules():
            if getattr(module, "persist", None) is self._bound:
                module.persist = self._real
        if exc_type is None and not self.calls and not self.allow_empty:
            raise AssertionError("PersistRecorder saw no persist call: the recorder is not wired to the code under test")
        return False
