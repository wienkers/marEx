"""Shared loaders and helpers for the pipeline-level detect tests.

Every loader returns a field rooted in a shipped zarr store (or built in memory), never a
``.persist()``ed one: a persisted input holds futures on whichever client was current when it
was built, and the tests fail with ``lost dependencies`` once that client closes.
``test_compute_mode_equivalence.py::test_loaders_carry_no_futures`` runs every loader here under
a live distributed client to keep it that way.
"""

from __future__ import annotations

import logging
from pathlib import Path

import dask
import numpy as np
import xarray as xr

from .._recorders import PersistRecorder  # noqa: F401  (re-exported for the detect tests)

DATA_DIR = Path(__file__).parent.parent / "data"

GRID_DIMS = {"time": "time", "x": "lon", "y": "lat"}
MESH_DIMS = {"time": "time", "x": "ncells"}
MESH_COORDS = {"time": "time", "x": "lon", "y": "lat"}

# The input the detect goldens were captured on: the last 7 years of the gridded fixture, with one
# masked cell. Must match the slice used when the goldens were regenerated.
N_GOLDEN_STEPS = 7 * 365

COMPARED_VARS = ("dat_anomaly", "extreme_events", "thresholds", "mask")


def open_store(name):
    """A shipped fixture store, lazily read and never persisted."""
    return xr.open_zarr(str(DATA_DIR / name), chunks={})


def open_sst_gridded():
    """The shipped gridded SST (40 years daily, 20 x 40 cells)."""
    return open_store("sst_gridded.zarr").to


def mask_one_cell(sst):
    """``sst`` with the cell at lat[1], lon[1] all-NaN (a land point)."""
    return sst.where(~((sst.lat == sst.lat[1]) & (sst.lon == sst.lon[1])), np.nan)


def load_golden_sst():
    """The detect goldens' input: the last ``N_GOLDEN_STEPS`` of the gridded fixture with one masked cell."""
    return mask_one_cell(open_sst_gridded().isel(time=slice(-N_GOLDEN_STEPS, None)))


def open_sst_mesh(lat_span=90.0, lon_span=180.0):
    """The shipped mesh SST (405 cells) with the mock ``lat``/``lon`` the pipeline requires.

    The fixture carries no coordinates; each caller keeps the span its tests were written with.
    """
    sst = open_store("sst_unstructured.zarr").to
    n = sst.sizes["ncells"]
    return sst.assign_coords(
        lat=xr.DataArray(np.linspace(-lat_span, lat_span, n), dims=["ncells"], name="lat"),
        lon=xr.DataArray(np.linspace(-lon_span, lon_span, n), dims=["ncells"], name="lon"),
    )


def materialise(ds, names=COMPARED_VARS):
    """The compared variables of ``ds`` as numpy arrays: a reference safe to share across tests."""
    return {name: np.asarray(ds[name].compute().values) for name in names if name in ds.data_vars}


def assert_identical(reference, candidate, label, names=COMPARED_VARS):
    """Every compared variable must match in dtype, shape, NaN pattern and value, with no tolerance.

    ``reference`` is a Dataset or a dict of numpy arrays from :func:`materialise`.
    """
    for name in names:
        ref_vars = reference if isinstance(reference, dict) else reference.data_vars
        if name not in ref_vars:
            continue
        assert name in candidate.data_vars, f"{label}: {name} missing from candidate"
        a = reference[name] if isinstance(reference, dict) else reference[name].compute().values
        b = candidate[name].compute().values
        assert a.dtype == b.dtype, f"{label}: {name} dtype moved {a.dtype} -> {b.dtype}"
        assert a.shape == b.shape, f"{label}: {name} shape moved {a.shape} -> {b.shape}"
        if a.dtype.kind == "f":
            nan_a, nan_b = np.isnan(a), np.isnan(b)
            assert np.array_equal(nan_a, nan_b), f"{label}: {name} NaN pattern differs"
            assert np.array_equal(a[~nan_a], b[~nan_b]), f"{label}: {name} is not bit-identical"
        else:
            assert np.array_equal(a, b), f"{label}: {name} is not bit-identical"


def check(failures, case, func):
    """Run one table row's assertions; record the first that fails, or the error the row raised, under its case id."""
    import pytest

    try:
        func()
    except (AssertionError, pytest.fail.Exception) as e:
        failures.append(f"{case}: {str(e).strip().splitlines()[0] if str(e).strip() else 'assertion failed'}")
    except Exception as e:  # noqa: BLE001 - a row that raises is a failing row; the others still run
        failures.append(f"{case}: raised {type(e).__name__}: {e}")


def on_client(client):
    """True when dask's current scheduler is ``client``."""
    return getattr(dask.base.get_scheduler(), "__self__", None) is client


# The pipeline's INFO output is emitted by whichever stage produced the line, so a capture that
# watches only the chainer misses the extremes stage's event-count summary.
PIPELINE_LOGGERS = ("marEx.pipeline", "marEx.anomaly.api", "marEx.extremes.api")


def capture_pipeline_log(fn, logger_names=PIPELINE_LOGGERS, level=logging.INFO):
    """Run ``fn`` and return ``(its result, the messages the pipeline's loggers emitted)``.

    marEx configures its own logging handler, so pytest's ``caplog`` captures nothing from it.
    A handler attached to the named loggers is independent of propagation and handler setup.
    """

    class _Collector(logging.Handler):
        def __init__(self):
            super().__init__(level=logging.DEBUG)
            self.messages = []

        def emit(self, record):
            self.messages.append(record.getMessage())

    collector = _Collector()
    loggers = [logging.getLogger(name) for name in logger_names]
    previous = [logger.level for logger in loggers]
    for logger in loggers:
        logger.addHandler(collector)
        logger.setLevel(level)
    try:
        result = fn()
    finally:
        for logger, level_before in zip(loggers, previous):
            logger.removeHandler(collector)
            logger.setLevel(level_before)
    return result, collector.messages
