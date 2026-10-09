"""
Every marEx zarr write must survive a format-2 input under zarr-python 3.

A variable opened from a format-2 store carries ``numcodecs`` codecs in ``.encoding``;
xarray re-applies them on ``to_zarr`` and zarr 3 refuses them in a format-3 store
("Expected a BytesBytesCodec"). The fixtures in ``tests/data`` are format-2 stores, so
under zarr 3 these tests fail on any write that skips ``write_zarr``. Under zarr 2 they
pass either way; the tripwire below is what holds there.
"""

import re
import shutil
import tempfile
from pathlib import Path

import pytest
import xarray as xr

import marEx
from marEx.core.encoding import STORE_ENCODING_KEYS, write_zarr
from marEx.track import morphology

DATA_DIR = Path(__file__).parent / "data"
PACKAGE_DIR = Path(marEx.__file__).parent


@pytest.fixture
def scratch():
    d = tempfile.mkdtemp(prefix="marex_test_store_enc_")
    yield d
    shutil.rmtree(d, ignore_errors=True)


@pytest.fixture(scope="module")
def extremes_gridded(dask_client):
    return xr.open_zarr(str(DATA_DIR / "extremes_gridded.zarr"), chunks={}).persist()


def _codec_keys(obj):
    return {name: sorted(k for k in v.encoding if k in STORE_ENCODING_KEYS) for name, v in obj.variables.items()}


def test_gridded_checkpoint_save_then_load(extremes_gridded, scratch, dask_client):
    """tracker.run_preprocess(checkpoint='save') writes the format-2 input's coords."""
    events = extremes_gridded.extreme_events.chunk({"time": 2, "lat": -1, "lon": -1})
    kwargs = {"area_filter_quartile": 0.5, "R_fill": 2, "T_fill": 2, "temp_dir": scratch, "quiet": True}

    saved, _ = marEx.tracker(events, extremes_gridded.mask, **kwargs).run_preprocess(checkpoint="save")
    loaded, _ = marEx.tracker(events, extremes_gridded.mask, **kwargs).run_preprocess(checkpoint="load")

    xr.testing.assert_identical(saved.load(), loaded.load())


@pytest.mark.parametrize(
    "store, chunks", [("extremes_gridded.zarr", {}), ("extremes_unstructured.zarr", {"time": 2, "ncells": -1})]
)
def test_refresh_dask_graph_on_format2_coords(store, chunks, scratch):
    """refresh_dask_graph's internal write (not reached with codecs by the public tracker on these fixtures)."""
    ex = xr.open_zarr(str(DATA_DIR / store), chunks=chunks)
    oid = xr.where(ex.extreme_events, 1, 0)

    refreshed = morphology.refresh_dask_graph(oid, f"{scratch}/refresh.zarr")

    assert (refreshed.values == oid.values).all()


def test_write_zarr_leaves_the_callers_encoding_alone(scratch):
    """The helper clears a shallow copy: the caller's variables keep their encoding."""
    ex = xr.open_zarr(str(DATA_DIR / "extremes_gridded.zarr"), chunks={})
    da = ex.extreme_events
    before = _codec_keys(da.to_dataset())
    assert any(before.values()), "fixture no longer carries store encoding; the test is vacuous"

    write_zarr(da, f"{scratch}/out.zarr", mode="w")

    assert _codec_keys(da.to_dataset()) == before


def test_every_to_zarr_goes_through_write_zarr():
    """Tripwire: a write that bypasses the helper is how three sites were missed twice."""
    offenders = []
    for path in PACKAGE_DIR.rglob("*.py"):
        if path.name == "encoding.py" and path.parent.name == "core":
            continue
        for lineno, line in enumerate(path.read_text().splitlines(), 1):
            code = line.split("#", 1)[0]
            if ">>>" in code or code.lstrip().startswith("... "):
                continue
            if re.search(r"\.to_zarr\(", code):
                offenders.append(f"{path.relative_to(PACKAGE_DIR)}:{lineno}: {line.strip()}")
    assert not offenders, "route these writes through marEx.core.encoding.write_zarr:\n" + "\n".join(offenders)
