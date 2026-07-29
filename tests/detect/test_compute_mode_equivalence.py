"""Cross-mode bit-identity: lazy and streaming must equal persist, byte for byte.

This is the equivalence test applied to the compute modes. Integer,
boolean and label outputs are compared with no tolerance, and so are the floats: no
compute mode reorders a reduction, so a float difference here is a bug, not rounding.

The modes differ only in *where* an intermediate lives -- cluster RAM, nowhere, or a
scratch zarr -- so any divergence points at a round-trip defect (a dtype demoted on write,
a coordinate lost, an encoding conflict) rather than at the numerics.

Each leg compares against a persist run on its own scheduler: the lazy and unstructured legs against one
built on the threaded scheduler, the gridded streaming legs against one built on the session client. Each is
computed once per combination and scheduler and shared, and is built on that scheduler explicitly, whichever
leg asks first, so it depends neither on test order nor on how xdist splits the module. It is materialised to
numpy (``conftest.materialise``) in a module-scoped fixture, never a ``.persist()``ed object, which would tie
it to whichever client was current when it was built; the getter asserts that. Requesting the client puts
every leg's test on it, so the lazy and unstructured legs pin themselves back to the threaded scheduler they
ran on before.
"""

import dask
import numpy as np
import pytest

import marEx
from marEx.core.compute_mode import clear_staging

from .conftest import (
    GRID_DIMS,
    MESH_COORDS,
    MESH_DIMS,
    assert_identical,
    load_golden_sst,
    materialise,
    on_client,
    open_sst_gridded,
    open_sst_mesh,
    open_store,
)


def test_loaders_carry_no_futures():
    """Re-adding ``.persist()`` to a detect loader pins every test's input to whichever client was current when
    it was built, and the tests fail with ``lost dependencies`` once that client closes.

    With no client current ``.persist()`` leaves no futures, so every loader is run here under its own live
    distributed client to make the check bite.
    """
    from distributed import Client
    from distributed.client import futures_of

    with Client(processes=False, n_workers=1, threads_per_worker=1, dashboard_address=None) as private:
        assert on_client(private), "the tripwire is not computing on its own client"
        loaders = {
            "open_sst_gridded()": open_sst_gridded,
            "load_golden_sst()": load_golden_sst,
            "open_sst_mesh()": open_sst_mesh,
            "open_sst_mesh(60, 180)": lambda: open_sst_mesh(60.0, 180.0),
            **{
                f"open_store({n!r})": (lambda n=n: open_store(n))
                for n in ("sst_gridded.zarr", "sst_unstructured.zarr", "extremes_unstructured.zarr")
            },
        }
        for label, load in loaders.items():
            held = futures_of(load())
            assert not held, f"{label} holds {len(held)} future(s); keep it rooted in the zarr store"


METHOD_COMBOS = [
    ("detrend_harmonic", "global_percentile"),
    ("detrend_harmonic", "seasonal_percentile"),
    ("shifting_baseline", "seasonal_percentile"),
    ("fixed_baseline", "global_percentile"),
    ("detrend_fixed_baseline", "seasonal_percentile"),
]


def _kwargs(method_anomaly, method_extreme, dimensions, coordinates=None):
    kw = {
        "method_anomaly": method_anomaly,
        "method_extreme": method_extreme,
        "threshold_percentile": 95,
        "dimensions": dimensions,
        "dask_chunks": {"time": 25},
    }
    if coordinates is not None:
        kw["coordinates"] = coordinates
    if method_anomaly == "shifting_baseline":
        kw.update(window_years=5, smooth_days=11)
    if method_extreme == "seasonal_percentile":
        kw.update(window_days=3)
    return kw


# The grid's inputs and keyword sets, by case.
def _case(key):
    if key == "mesh":
        return open_sst_mesh(), _kwargs("detrend_harmonic", "seasonal_percentile", MESH_DIMS, MESH_COORDS)
    return open_sst_gridded(), _kwargs(*key, GRID_DIMS)


# What ``tests/conftest.py`` pins a test to when it does not request the session client.
THREADED = {"scheduler": "threads", "array.rechunk.method": "tasks"}


@pytest.fixture(scope="module")
def reference(client):
    """``reference(key, scheduler)``: the persist run of ``key`` on ``scheduler`` ("threads" or "client"), computed
    once per module and shared by the legs that run on that scheduler.

    Module-scoped, so the arrays are released before the module's leak check; numpy, so they hold no futures.
    """
    cache = {}
    # The scheduler and rechunk method a test computes with: pinned to threads, or on the session client.
    schedulers = {"threads": THREADED, "client": {"scheduler": client, "array.rechunk.method": None}}

    def get(key, scheduler):
        if (key, scheduler) not in cache:
            sst, kw = _case(key)
            with dask.config.set(schedulers[scheduler]):
                built_on_client = on_client(client)
                assert built_on_client == (scheduler == "client"), f"reference {key} is not being built on {scheduler}"
                cache[key, scheduler] = materialise(marEx.preprocess_data(sst, compute_mode="persist", **kw))
        ref = cache[key, scheduler]
        for name, values in ref.items():
            if not isinstance(values, np.ndarray):
                kind = type(values).__name__
                # The failure's traceback keeps this frame alive; drop the entry and the names first, or a
                # persisted reference would outlive the module and fail its leak check as well.
                del cache[key, scheduler], ref, values
                raise AssertionError(f"shared reference {key}/{name} is a {kind}, not numpy")
        return ref

    yield get
    cache.clear()


class TestLazyEquivalence:
    """lazy must reproduce persist exactly, across every method combination."""

    @pytest.mark.parametrize("method_anomaly,method_extreme", METHOD_COMBOS)
    def test_lazy_matches_persist(self, method_anomaly, method_extreme, reference):
        key = (method_anomaly, method_extreme)
        ref = reference(key, "threads")
        with dask.config.set(THREADED):
            sst, kw = _case(key)
            lazy = marEx.preprocess_data(sst, compute_mode="lazy", **kw)
            assert_identical(ref, lazy, f"lazy/{method_anomaly}/{method_extreme}")


class TestStreamingEquivalence:
    """streaming must reproduce persist exactly, across every method combination, on the session client's workers."""

    @pytest.mark.parametrize("method_anomaly,method_extreme", METHOD_COMBOS)
    def test_streaming_matches_persist(self, method_anomaly, method_extreme, tmp_path, client, reference):
        assert on_client(client), "not computing on the session client"
        key = (method_anomaly, method_extreme)
        sst, kw = _case(key)
        streamed = marEx.preprocess_data(sst, compute_mode="streaming", scratch_dir=str(tmp_path), **kw)
        try:
            assert_identical(reference(key, "client"), streamed, f"streaming/{method_anomaly}/{method_extreme}")
        finally:
            clear_staging(streamed)


class TestUnstructuredEquivalence:
    """The unstructured path has its own coordinate handling; round-trip it too."""

    def test_lazy_matches_persist(self, reference):
        ref = reference("mesh", "threads")
        with dask.config.set(THREADED):
            sst, kw = _case("mesh")
            lazy = marEx.preprocess_data(sst, compute_mode="lazy", **kw)
            assert_identical(ref, lazy, "lazy/unstructured")

    def test_streaming_matches_persist(self, tmp_path, reference):
        ref = reference("mesh", "threads")
        with dask.config.set(THREADED):
            sst, kw = _case("mesh")
            streamed = marEx.preprocess_data(sst, compute_mode="streaming", scratch_dir=str(tmp_path), **kw)
            try:
                assert_identical(ref, streamed, "streaming/unstructured")
            finally:
                clear_staging(streamed)
