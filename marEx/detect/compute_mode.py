"""
Materialisation policy for the marEx detection pipeline.

:func:`marEx.preprocess_data` has three honest positions on a trilemma -- low RAM, low
recompute, low disk I/O; pick two -- and this module is where that choice is made concrete.
Every materialisation site in ``detect`` routes through a :class:`Materialiser`, so the
modes are three implementations of two verbs rather than conditionals scattered through the
pipeline.

Two verbs, because the pinned intermediates fall into two distinct scaling classes:

``pin``
    Bounded or space-scaled intermediates whose only job is to stop a re-computation
    within one stage. Materialised in ``persist`` mode, skipped otherwise.

``stage``
    Anchors: objects read by two or more downstream consumers, where leaving them lazy
    re-executes the entire upstream graph once per consumer. ``persist`` mode pins them in
    cluster RAM; ``streaming`` mode writes them to a scratch zarr and re-opens them, which
    is the same anchoring guarantee with a disk-resident rather than RAM-resident backing
    store; ``lazy`` mode returns them untouched and accepts the recompute.

Streaming mode is conceptually the checkpointing that was removed from marEx earlier this
year. It was removed because the *implementation* was a set of hacks -- fixed scratch paths
and tuple bandages -- not because the idea was wrong, so it returns here as a first-class,
explicit, opt-in mode with per-run unique paths.

This module is a leaf in the detect dependency graph: it imports only dask, xarray, and the
package logger and exceptions.
"""

import atexit
import os
import shutil
import uuid
from pathlib import Path
from typing import Any, Literal, Optional, Tuple, Union

import dask
import xarray as xr

from ..exceptions import ConfigurationError
from ..logging_config import get_logger

logger = get_logger(__name__)

ComputeMode = Literal["persist", "lazy", "streaming"]

VALID_COMPUTE_MODES = ("persist", "lazy", "streaming")

# Staging directories created by this process, cleaned at interpreter exit as a backstop
# against littering scratch. Deliberately NOT cleaned when preprocess_data returns: in
# streaming mode the returned Dataset is lazy and reads from these stores, so removing
# them before the caller has written its output would break the result.
_STAGING_DIRS: list = []


def create_staging_dir(scratch_dir: Union[str, Path]) -> Path:
    """
    Create a unique per-run staging directory under ``scratch_dir``.

    Parameters
    ----------
    scratch_dir : str or pathlib.Path
        Parent directory, created if it does not exist. Should be large, transient
        storage (e.g. a scratch filesystem), not a home directory.

    Returns
    -------
    pathlib.Path
        The newly created, empty staging directory.

    Notes
    -----
    Uniqueness is per (process, call). The checkpointing removed from marEx earlier this
    year used fixed scratch paths, so two concurrent runs collided silently; the same
    lesson was already forced on the tracker's temporary stores.
    """
    root = Path(scratch_dir)
    root.mkdir(parents=True, exist_ok=True)
    path = root / f"marex_stage_{os.getpid()}_{uuid.uuid4().hex[:8]}"
    path.mkdir(parents=False, exist_ok=False)
    _STAGING_DIRS.append(path)
    logger.info(f"Streaming staging directory: {path}")
    return path


def clear_staging(target: Union[str, Path, xr.Dataset]) -> None:
    """
    Remove a staging directory created by ``compute_mode='streaming'``.

    Parameters
    ----------
    target : str or pathlib.Path or xarray.Dataset
        The staging directory, or a Dataset returned by streaming mode (its
        ``marex_staging_dir`` attribute is used).

    Notes
    -----
    Idempotent: clearing an already-removed directory is not an error. Call this only
    after the returned Dataset has been written out, since in streaming mode it reads
    lazily from the staged stores.
    """
    if isinstance(target, xr.Dataset):
        recorded = target.attrs.get("marex_staging_dir")
        if not recorded:
            logger.debug("Dataset carries no marex_staging_dir attribute; nothing to clear")
            return
        path = Path(recorded)
    else:
        path = Path(target)

    if path.exists():
        shutil.rmtree(path, ignore_errors=True)
        logger.debug(f"Cleared staging directory: {path}")
    if path in _STAGING_DIRS:
        _STAGING_DIRS.remove(path)


@atexit.register
def _clear_all_staging() -> None:  # pragma: no cover - runs at interpreter shutdown
    """Remove any staging directories this process created but did not clear."""
    for path in list(_STAGING_DIRS):
        clear_staging(path)


class Materialiser:
    """
    Decide how a pipeline intermediate is materialised, per :data:`ComputeMode`.

    Parameters
    ----------
    mode : {'persist', 'lazy', 'streaming'}
        Materialisation policy. See the module docstring for the trilemma.
    staging_dir : pathlib.Path, optional
        Directory for staged zarr stores. Required when ``mode='streaming'``; ignored
        otherwise. Use :func:`create_staging_dir` to build one.

    Raises
    ------
    ConfigurationError
        If ``mode`` is not a valid compute mode, or if ``mode='streaming'`` without a
        ``staging_dir``.

    Examples
    --------
    >>> m = Materialiser("lazy")
    >>> m.is_lazy
    True
    """

    def __init__(self, mode: ComputeMode, staging_dir: Optional[Path] = None) -> None:
        """Validate the mode and record it with its staging directory."""
        if mode not in VALID_COMPUTE_MODES:
            raise ConfigurationError(
                f"Unknown compute_mode '{mode}'",
                details="Invalid compute_mode parameter",
                suggestions=[
                    "Use 'persist' (default) when the data fits in aggregate cluster RAM",
                    "Use 'lazy' to return an unmaterialised Dataset, accepting recompute per consumer",
                    "Use 'streaming' with scratch_dir set for inputs larger than cluster RAM",
                ],
                context={"provided_mode": str(mode), "valid_modes": list(VALID_COMPUTE_MODES)},
            )
        if mode == "streaming" and staging_dir is None:
            raise ConfigurationError(
                "compute_mode='streaming' requires scratch_dir",
                details="Streaming mode stages shared intermediates to a zarr store on disk",
                suggestions=[
                    "Pass scratch_dir pointing at large transient storage, e.g. a scratch filesystem",
                    "Use compute_mode='lazy' if no scratch space is available",
                ],
                context={"provided_mode": str(mode)},
            )
        self.mode: ComputeMode = mode
        self.staging_dir = staging_dir
        # Arrays this Materialiser has written to a staging store, keyed by the name they
        # were staged under. Holds a strong reference deliberately: `stage`'s already-staged
        # check is an identity test, and a dead weakref would let id() be reused. The values
        # are lazy zarr-backed arrays (a task graph, not data), so the cost is negligible.
        self._staged: dict = {}

    def __repr__(self) -> str:
        """Return an unambiguous representation naming the mode and staging directory."""
        return f"Materialiser(mode={self.mode!r}, staging_dir={self.staging_dir!r})"

    @property
    def is_lazy(self) -> bool:
        """bool: True when this mode leaves intermediates unmaterialised in cluster RAM."""
        return self.mode != "persist"

    @property
    def is_streaming(self) -> bool:
        """True when intermediates are staged to disk rather than pinned in RAM."""
        return self.mode == "streaming"

    def pin(self, *objs: Any) -> Tuple[Any, ...]:
        """
        Materialise bounded intermediates. A no-op outside ``persist`` mode.

        Parameters
        ----------
        *objs
            Dask collections (or xarray objects wrapping them).

        Returns
        -------
        tuple
            The objects, materialised in ``persist`` mode and untouched otherwise.
        """
        if self.mode == "persist":
            return dask.persist(*objs)
        return objs

    def pin_one(self, obj: Any) -> Any:
        """
        :meth:`pin` for a single object.

        Parameters
        ----------
        obj
            A dask collection, or an xarray object wrapping one.

        Returns
        -------
        object
            The object, materialised in ``persist`` mode and untouched otherwise.
        """
        (out,) = self.pin(obj)
        return out

    def stage(self, obj: xr.DataArray, label: str, preserve_chunks: bool = False) -> xr.DataArray:
        """
        Anchor an object that several downstream consumers read.

        Parameters
        ----------
        obj : xarray.DataArray
            The array to anchor.
        label : str
            Short identifier, used as the staged store's filename in streaming mode.
        preserve_chunks : bool, default False
            In ``streaming`` mode only, restore ``obj``'s chunk layout after staging if
            ``_stage_to_zarr``'s uniform rechunk changed it. Default ``False`` leaves
            detect's existing, documented behaviour (staging may change chunk layout)
            completely unchanged; callers with downstream logic that depends on chunk
            boundaries not moving (e.g. the tracker's merge loop) should pass ``True``.

        Returns
        -------
        xarray.DataArray
            In ``persist`` mode the RAM-pinned array; in ``streaming`` mode a lazy array
            re-opened from a scratch zarr; in ``lazy`` mode ``obj`` unchanged.
        """
        if self.mode == "persist":
            return obj.persist()
        if self.mode == "lazy":
            return obj
        if self._staged.get(label) is obj:
            # `obj` IS the array this Materialiser wrote to `<label>.zarr` and re-opened.
            # It is already disk-anchored, so re-staging would re-run the graph that
            # reads the store only to write the same bytes back over it. Identity, not
            # equality: a derived array must still be staged on its own.
            return obj
        self._reject_relabel(label, obj)
        staged = self._stage_to_zarr(obj, label)
        if preserve_chunks and obj.chunks is not None and staged.chunks is not None and staged.chunks != obj.chunks:
            staged = staged.chunk(dict(zip(obj.dims, obj.chunks)))
        self._staged[label] = staged
        return staged

    def _reject_relabel(self, label: str, obj: Any) -> None:
        """
        Refuse to stage a *different* array under a name already in use.

        Staging writes ``<staging_dir>/<label>.zarr`` with ``mode="w"``. Doing that twice
        under one label silently rewrites the store that the FIRST staged array is still
        lazily reading, so that array starts returning the second one's bytes -- wrong
        answers, no error. The registry makes this worse rather than better, because it
        holds a strong reference that keeps the stale array alive to be read.

        No caller does this today (every label is staged at exactly one site, and the two
        `thresholds` sites are mutually exclusive methods). The hazard is real for any
        caller that stages inside a loop -- e.g. the unstructured merge loop, which would
        want a per-iteration array. Such a caller must pass a per-iteration label; this
        turns the silent corruption into an immediate, explanatory failure.
        """
        previous = self._staged.get(label)
        if previous is None or previous is obj:
            return
        raise ConfigurationError(
            f"Materialiser.stage({label!r}) called with a different array than the one already staged",
            details=(
                "Staging writes <staging_dir>/<label>.zarr with mode='w'. Re-using a label for a "
                "different array overwrites the store the earlier array still reads from, which "
                "silently changes its values rather than raising."
            ),
            suggestions=[
                "Give each staged array its own label, e.g. append a loop iteration index: "
                f"'{label}_iter0', '{label}_iter1', ...",
                "If the earlier array is genuinely dead, drop it from the registry before re-staging",
            ],
            context={"label": label},
        )

    def stage_many(self, objs: dict, label: str, preserve_chunks: bool = False) -> Tuple[xr.DataArray, ...]:
        """
        Anchor several arrays that **one graph** produces, in a single materialisation.

        :meth:`stage` called once per array would execute that shared graph once per
        array. The motivating case is ``marEx.track.morphology``'s absolute-threshold
        filter, where a single per-slice labelling pass yields both the keep-mask and the
        area census: staging them separately labels every slice twice.

        Parameters
        ----------
        objs : dict of str to xarray.DataArray
            Arrays to anchor together, keyed by the variable name to store them under.
        label : str
            Short identifier for the shared store, and the key each array is registered
            under for :meth:`stage`'s already-staged check.
        preserve_chunks : bool, default False
            As :meth:`stage`, applied to each array independently.

        Returns
        -------
        tuple of xarray.DataArray
            The arrays in ``objs`` order, anchored per the mode.
        """
        if self.mode == "persist":
            return dask.persist(*objs.values())
        if self.mode == "lazy":
            return tuple(objs.values())

        logger.info(f"Staging {list(objs)} jointly as '{label}'")
        reopened = self._write_and_reopen(xr.Dataset({name: obj.rename(name) for name, obj in objs.items()}), label)

        out = []
        for name, source in objs.items():
            staged = self._restore(reopened[name], source)
            if preserve_chunks and source.chunks is not None and staged.chunks is not None and staged.chunks != source.chunks:
                staged = staged.chunk(dict(zip(source.dims, source.chunks)))
            # Registered under the VARIABLE name, not `label`: the caller that re-anchors
            # one of these (the tracker re-staging data_bin_filtered) names the array, not
            # the joint store, so that call short-circuits instead of re-writing.
            self._reject_relabel(name, staged)
            self._staged[name] = staged
            out.append(staged)
        return tuple(out)

    def _stage_to_zarr(self, obj: xr.DataArray, label: str) -> xr.DataArray:
        """
        Write ``obj`` to a zarr store under the staging directory and re-open it.

        Re-opening with ``chunks={}`` restores the on-disk chunking, which is the dask
        chunking the array was written with, so downstream chunk assumptions still hold.
        """
        if self.staging_dir is None:  # pragma: no cover - guaranteed by __init__
            raise ConfigurationError(
                "Streaming materialiser has no staging directory",
                details="Internal invariant violated: staging_dir is None in streaming mode",
                suggestions=["Construct the Materialiser via preprocess_data(compute_mode='streaming', scratch_dir=...)"],
            )

        name = obj.name or label
        logger.info(f"Staging '{label}' to {self.staging_dir / f'{label}.zarr'}")
        return self._restore(self._write_and_reopen(obj.to_dataset(name=name), label)[name], obj)

    def _write_and_reopen(self, ds: xr.Dataset, label: str) -> xr.Dataset:
        """
        Write ``ds`` to ``<staging_dir>/<label>.zarr`` and re-open it lazily.

        Re-opening with ``chunks={}`` restores the on-disk chunking, which is the dask
        chunking the dataset was written with, so downstream chunk assumptions still hold.
        Writing every variable in ONE call is what makes :meth:`stage_many` execute a
        shared upstream graph once rather than once per variable.
        """
        if self.staging_dir is None:  # pragma: no cover - guaranteed by __init__
            raise ConfigurationError(
                "Streaming materialiser has no staging directory",
                details="Internal invariant violated: staging_dir is None in streaming mode",
                suggestions=["Construct the Materialiser via preprocess_data(compute_mode='streaming', scratch_dir=...)"],
            )
        path = self.staging_dir / f"{label}.zarr"

        # Stale `chunks` encoding carried in from an upstream open_zarr conflicts with the
        # dask chunking now being written. The pipeline clears this on its own output for
        # exactly the same reason.
        for var in list(ds.variables):
            ds[var].encoding.pop("chunks", None)

        # Zarr requires uniform chunk sizes (a smaller FINAL chunk is allowed). Several
        # anomaly methods leave ragged chunking behind -- the fixed_baseline groupby
        # produces e.g. (30, 30, ..., 6, 24, 30, ...) along time -- and `to_zarr` rejects
        # that outright. `persist` mode never hits it because it never writes to disk, so
        # this is a failure mode unique to staging. Rechunking to the largest chunk per
        # dimension is value-neutral: it changes task granularity only, and the array is
        # re-opened with the on-disk chunking immediately below anyway.
        uniform: dict = {}
        for variable in ds.variables.values():
            if variable.chunks is None:
                continue
            for dim, sizes in zip(variable.dims, variable.chunks):
                uniform[dim] = max(uniform.get(dim, 0), max(sizes))
        if uniform:
            ds = ds.chunk(uniform)

        ds.to_zarr(path, mode="w", consolidated=True)
        return xr.open_zarr(path, consolidated=True, chunks={})

    @staticmethod
    def _restore(reopened: xr.DataArray, obj: xr.DataArray) -> xr.DataArray:
        """Make a re-opened array substitutable for the one that was staged."""
        # Drop the round-tripped encoding: re-writing a coordinate that arrived with CF
        # encoding (notably `time`) otherwise conflicts when the caller saves the output.
        for coord in reopened.coords:
            reopened[coord].encoding = {}
        reopened.encoding = {}

        # Restore any non-dimension coordinates zarr did not carry through, so the staged
        # array is substitutable for the original everywhere downstream.
        for cname, cvar in obj.coords.items():
            if cname not in reopened.coords:
                reopened = reopened.assign_coords({cname: cvar})

        reopened.attrs.update(obj.attrs)

        # Restore the original name unconditionally. The store's variable name falls back
        # to the label when the array is unnamed, so an unnamed input would otherwise come
        # back named "thresholds"/"dat_anomaly" -- a difference between streaming and the
        # other two modes, since `persist` and `lazy` both preserve `name=None`. xarray's
        # binary ops keep a name only when both operands agree, so a spurious name here can
        # silently change the name of a downstream comparison result.
        reopened.name = obj.name
        return reopened
