"""Every method combination on both grids yields a well-formed, correctly sized, self-describing dataset.

This is a smoke matrix, not a numerics check (those are the goldens and the equivalence suite):
each row checks the output's variables, dtypes, dims, method attributes, time length, the
mask's own attributes, and that the extreme frequency sits near the requested percentile.
Rows of one grid and percentile method share a test; every row runs, failures are collected
with the row's case id, and the test fails once at the end.

The input is the last ``N_MATRIX_STEPS`` (15 years) of each shipped fixture, not the 40-year record.
The shifting baseline needs about that much: its first output years see few prior years, and on
a 10-year input (5 output years) its seasonal frequency fell to 0.036-0.039, out of the band
(MEASURED, pins/detect/pins_matrix*.out); 15 years gave 0.048. It carries an SST's
attributes (``INPUT_ATTRS``), so every row also checks that the output ``mask`` describes itself
and not the input field.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.anomaly.api import _anomaly_steps
from marEx.exceptions import ConfigurationError
from marEx.extremes.api import _extreme_steps

from ..conftest import assert_no_failures, assert_percentile_frequency
from .conftest import DATA_DIR, GRID_DIMS, MESH_COORDS, MESH_DIMS, check, mask_one_cell, on_client, open_sst_gridded, open_sst_mesh

N_MATRIX_STEPS = 15 * 365
INPUT_ATTRS = {"standard_name": "sea_surface_temperature", "units": "K", "valid_min": 250.0}
DASK_CHUNKS = {"time": 25}
ALL_COMBOS = [
    ("fixed_baseline", "global_percentile"),
    ("fixed_baseline", "seasonal_percentile"),
    ("detrend_fixed_baseline", "global_percentile"),
    ("detrend_fixed_baseline", "seasonal_percentile"),
    ("shifting_baseline", "global_percentile"),
    ("shifting_baseline", "seasonal_percentile"),
    ("detrend_harmonic", "global_percentile"),
    ("detrend_harmonic", "seasonal_percentile"),
]


def gridded_input():
    """The last 15 years of the gridded fixture, the cell at lat[1], lon[1] masked, carrying SST attrs."""
    return mask_one_cell(open_sst_gridded().isel(time=slice(-N_MATRIX_STEPS, None))).assign_attrs(INPUT_ATTRS)


def mesh_input():
    """The last 15 years of the mesh fixture, cell 2 masked, carrying SST attrs."""
    sst = open_sst_mesh().isel(time=slice(-N_MATRIX_STEPS, None))
    return sst.where(~(sst.ncells == 2), np.nan).assign_attrs(INPUT_ATTRS)


def mesh_attachments(ncells):
    """Mock neighbours (seeded) and cell areas; real ones come from the grid files."""
    neighbours = xr.DataArray(np.random.default_rng(0).integers(0, ncells, (3, ncells)), dims=["nv", "ncells"])
    cell_areas = xr.DataArray(np.ones(ncells) * 1000.0, dims=["ncells"])
    return neighbours, cell_areas


def _method_kwargs(method_anomaly, method_extreme, grid, method_percentile):
    """Each combination's keyword set. ``detrend_orders`` is passed explicitly where the method reads it."""
    kw = {"method_anomaly": method_anomaly, "method_extreme": method_extreme, "threshold_percentile": 95}
    if method_percentile == "exact":
        kw["method_percentile"] = "exact"
    if method_anomaly == "shifting_baseline":
        kw.update(window_years=5, smooth_days=5 if (grid == "mesh" or method_percentile == "exact") else 11)
    if method_anomaly in ("detrend_harmonic", "detrend_fixed_baseline"):
        kw["detrend_orders"] = [1, 2]
    if method_anomaly == "detrend_fixed_baseline":
        kw["force_zero_mean"] = True
    if method_extreme == "seasonal_percentile":
        # 11 days on every row. Shorter windows on the 15-year input put the frequency out of the band
        # (MEASURED, pins/detect/pins_matrix.out and pins_exact_shifting.out): mesh 5 days 0.065, exact
        # shifting 3 days 0.073 (0.057 on the 40-year record); 11 days gave 0.048-0.055.
        kw["window_days"] = 11
    return kw


def _rows(case):
    """``{row id: (preprocess_data kwargs, extra checks)}`` for one matrix case."""
    grid, method_percentile = case.split("-")
    if case == "gridded-exact":
        combos = [
            ("shifting_baseline", "seasonal_percentile"),
            ("detrend_harmonic", "global_percentile"),
            ("detrend_harmonic", "seasonal_percentile"),
        ]
    else:
        combos = ALL_COMBOS
    rows = {f"{a}/{e}": _method_kwargs(a, e, grid, method_percentile) for a, e in combos}
    if case == "mesh-approximate":
        # Without attachments, at a 50-step time chunk: the p90 rows on both threshold methods, and the
        # p95 global row that exercises the unstructured histogram rechunk.
        for method_extreme in ("global_percentile", "seasonal_percentile"):
            kw = {
                "method_anomaly": "detrend_harmonic",
                "method_extreme": method_extreme,
                "threshold_percentile": 90,
                "dask_chunks": {"time": 50},
            }
            if method_extreme == "seasonal_percentile":
                kw["window_days"] = 5
            rows[f"p90/detrend_harmonic/{method_extreme}/no-attachments"] = kw
        rows["rechunk/detrend_harmonic/global_percentile/no-attachments"] = {
            "method_anomaly": "detrend_harmonic",
            "method_extreme": "global_percentile",
            "threshold_percentile": 95,
            "dask_chunks": {"time": 50},
        }
    return rows


def _check_row(ds, sst, kw, grid, mask_shapes):
    method_anomaly, method_extreme = kw["method_anomaly"], kw["method_extreme"]
    percentile = kw["threshold_percentile"]
    time_dim = "time"
    cells = ("ncells",) if grid == "mesh" else ("lat", "lon")

    assert isinstance(ds, xr.Dataset)
    for var in ("extreme_events", "dat_anomaly", "thresholds", "mask"):
        assert var in ds.data_vars, f"missing {var}"
    assert ds.attrs["method_anomaly"] == method_anomaly
    assert ds.attrs["method_extreme"] == method_extreme
    assert ds.attrs["threshold_percentile"] == percentile
    if kw.get("method_percentile") == "exact":
        assert ds.attrs["method_percentile"] == "exact"
    assert ds.extreme_events.dtype == bool
    assert ds.dat_anomaly.dtype == np.float32

    assert set(ds.extreme_events.dims) == {time_dim, *cells}, ds.extreme_events.dims
    if method_extreme == "seasonal_percentile":
        assert "dayofyear" in ds.thresholds.dims, ds.thresholds.dims
    else:
        assert "dayofyear" not in ds.thresholds.dims, ds.thresholds.dims
    assert set(cells) <= set(ds.thresholds.dims), ds.thresholds.dims

    n_in, n_out = sst.sizes[time_dim], ds.sizes[time_dim]
    if method_anomaly == "shifting_baseline":
        reduction = n_in - n_out
        assert abs(reduction - kw["window_years"] * 365) <= 10, f"time reduced by {reduction}, expected ~{kw['window_years'] * 365}"
    else:
        assert n_out == n_in, f"time {n_out} != input {n_in}"

    if grid == "mesh":
        assert "lat" not in ds.extreme_events.dims and "lon" not in ds.extreme_events.dims
        assert "quantile" not in ds.extreme_events.coords
        if "neighbours" in kw:
            assert "neighbours" in ds.data_vars and "cell_areas" in ds.data_vars
            assert ds.neighbours.shape == kw["neighbours"].shape and ds.cell_areas.shape == kw["cell_areas"].shape

    # The output mask describes itself, not the input field it was derived from.
    assert not set(INPUT_ATTRS) & set(ds.mask.attrs), ds.mask.attrs
    assert ds.mask.attrs["long_name"] == "valid-data mask"
    mask_shapes.add(ds.mask.shape)

    if method_anomaly == "detrend_fixed_baseline":
        assert ds.attrs["detrend_orders"] == [1, 2]
        # Boolean flags are stored as int8 (0/1) so the dataset stays NetCDF-saveable.
        assert bool(ds.attrs["force_zero_mean"]) is True
        anomaly_mean = float(ds.dat_anomaly.mean())
        assert abs(anomaly_mean) < 0.01, f"force_zero_mean: anomaly mean {anomaly_mean}"

    if kw.get("method_percentile") == "exact" and method_anomaly == "detrend_harmonic" and method_extreme == "seasonal_percentile":
        assert ds.thresholds.dims == ("dayofyear", "lat", "lon"), ds.thresholds.dims
        assert int(ds.extreme_events.sum()) > 0
        thresholds = ds.thresholds.values
        finite = np.isfinite(thresholds)
        assert finite.sum() > 0 and np.all(thresholds[finite] > 0), "exact seasonal thresholds are not finite and positive"

    frequency = float(ds.extreme_events.mean())
    # tolerance_std only binds when a sample size is given; kept from the exact global row's original.
    tolerance = 1.0 if kw.get("method_percentile") == "exact" and method_extreme == "global_percentile" else 2.0
    assert_percentile_frequency(frequency, percentile, tolerance_std=tolerance, description=f"{method_anomaly} + {method_extreme}")


@pytest.mark.parametrize("case", ["gridded-approximate", "gridded-exact", "mesh-approximate"])
def test_method_matrix(case):
    grid = case.split("-")[0]
    if grid == "mesh":
        sst = mesh_input()
        neighbours, cell_areas = mesh_attachments(sst.sizes["ncells"])
        base = {"dimensions": MESH_DIMS, "coordinates": MESH_COORDS, "dask_chunks": DASK_CHUNKS}
    else:
        sst = gridded_input()
        base = {"dimensions": GRID_DIMS, "dask_chunks": DASK_CHUNKS}

    failures, mask_shapes = [], set()
    for row_id, kw in _rows(case).items():
        if grid == "mesh" and "no-attachments" not in row_id:
            kw = {**kw, "neighbours": neighbours, "cell_areas": cell_areas}

        def row(kw=kw):
            ds = marEx.preprocess_data(sst, **{**base, **kw})
            _check_row(ds, sst, kw, grid, mask_shapes)

        check(failures, row_id, row)
    assert_no_failures(failures)
    assert len(mask_shapes) == 1, f"rows disagree on the mask shape: {mask_shapes}"


@pytest.mark.parametrize("grid", ["gridded", "mesh"])
def test_custom_dimension_names(grid):
    """Custom dimension and coordinate names reach the output, under both threshold methods."""
    if grid == "gridded":
        sst = gridded_input()
        da = xr.DataArray(
            sst.values,
            dims=["t", "y", "x"],
            coords={"T": ("t", sst.time.values), "latitude": ("y", sst.lat.values), "longitude": ("x", sst.lon.values)},
            attrs=sst.attrs,
        ).chunk({"t": 25})
        dimensions = {"time": "t", "x": "x", "y": "y"}
        coordinates = {"time": "T", "x": "longitude", "y": "latitude"}
        cells, extra = ("y", "x"), {}
    else:
        sst = mesh_input()
        neighbours, cell_areas = mesh_attachments(sst.sizes["ncells"])
        da = sst.rename({"time": "t", "ncells": "cell", "lon": "longitude", "lat": "latitude"})
        dimensions = {"time": "t", "x": "cell"}
        coordinates = {"time": "t", "x": "longitude", "y": "latitude"}
        cells = ("cell",)
        extra = {"neighbours": neighbours.rename({"ncells": "cell"}), "cell_areas": cell_areas.rename({"ncells": "cell"})}

    common = {"threshold_percentile": 95, "dimensions": dimensions, "coordinates": coordinates, "dask_chunks": {"t": 25}, **extra}
    detrended = marEx.preprocess_data(
        da, method_anomaly="detrend_harmonic", method_extreme="global_percentile", detrend_orders=[1, 2], **common
    )
    shifting = marEx.preprocess_data(
        da,
        method_anomaly="shifting_baseline",
        method_extreme="seasonal_percentile",
        window_years=5,
        smooth_days=11,
        window_days=11,  # see _method_kwargs: on the mesh a 5-day window measured 0.066 on this input
        **common,
    )

    for label, ds, method_extreme in [("detrended", detrended, "global_percentile"), ("shifting", shifting, "seasonal_percentile")]:
        for var in ("extreme_events", "dat_anomaly", "thresholds", "mask"):
            assert var in ds.data_vars, f"{label}: missing {var}"
        assert "t" in ds.extreme_events.dims and all(c in ds.extreme_events.dims for c in cells), (label, ds.extreme_events.dims)
        for coord in ("latitude", "longitude") + (("T",) if grid == "gridded" else ()):
            assert coord in ds.coords, f"{label}: coordinate {coord} missing"
        assert ds.attrs["method_extreme"] == method_extreme
        assert ("dayofyear" in ds.thresholds.dims) == (method_extreme == "seasonal_percentile"), (label, ds.thresholds.dims)
        assert all(c in ds.thresholds.dims for c in cells), (label, ds.thresholds.dims)
        if grid == "mesh":
            assert "neighbours" in ds.data_vars and "cell_areas" in ds.data_vars
            assert "latitude" not in ds.extreme_events.dims and "longitude" not in ds.extreme_events.dims
            assert ds.latitude.dims == ("cell",) and ds.longitude.dims == ("cell",)
        assert_percentile_frequency(float(ds.extreme_events.mean()), 95, description=f"custom dimensions ({grid}): {label}")
    assert detrended.attrs["method_anomaly"] == "detrend_harmonic"
    assert shifting.attrs["method_anomaly"] == "shifting_baseline"
    assert shifting.sizes["t"] < da.sizes["t"], "shifting_baseline should reduce the time dimension"
    if grid == "mesh":
        assert detrended.neighbours.shape == shifting.neighbours.shape
        assert detrended.cell_areas.shape == shifting.cell_areas.shape


def test_auxiliary_coords_survive_save_roundtrip(client, tmp_path):
    """Auxiliary (non-index) coordinates must survive a distributed save round trip, to zarr AND netCDF.

    Non-index coords (integer grid indices ``x``/``y`` carried alongside lon/lat) become dask-backed
    via ``.chunk()``. A "tuple"/store bug under an older dask once made this fail, which the code
    worked around by eagerly materialising every coordinate; that workaround is gone, so this
    guards the actual save path on the session client's workers. Two years suffice.
    """
    assert on_client(client), "not computing on the session client"
    sst = gridded_input().isel(time=slice(-2 * 365, None))
    sst_aux = sst.assign_coords(
        x=("lon", np.arange(sst.sizes["lon"], dtype="int32")),
        y=("lat", np.arange(sst.sizes["lat"], dtype="int32")),
    )
    ds = marEx.preprocess_data(
        sst_aux,
        method_anomaly="detrend_harmonic",
        method_extreme="global_percentile",
        threshold_percentile=95,
        detrend_orders=[1, 2],
        dimensions=GRID_DIMS,
        dask_chunks=DASK_CHUNKS,
    )
    assert "x" in ds.coords and "y" in ds.coords

    for fmt in ("zarr", "netcdf"):
        if fmt == "zarr":
            target = tmp_path / "aux.zarr"
            ds.to_zarr(target, mode="w")
            reloaded = xr.open_zarr(target)
        else:
            target = tmp_path / "aux.nc"
            ds.to_netcdf(target)
            reloaded = xr.open_dataset(target)
        np.testing.assert_array_equal(reloaded["x"].values, np.arange(sst.sizes["lon"]), err_msg=fmt)
        np.testing.assert_array_equal(reloaded["y"].values, np.arange(sst.sizes["lat"]), err_msg=fmt)


def test_standardise_adds_the_stn_fields():
    """``standardise=True`` adds dat_stn, STD, extreme_events_stn and thresholds_stn, each well formed."""
    sst = gridded_input()
    ds = marEx.preprocess_data(
        sst,
        method_anomaly="detrend_harmonic",
        method_extreme="global_percentile",
        threshold_percentile=95,
        standardise=True,
        detrend_orders=[1, 2],
        dimensions=GRID_DIMS,
        dask_chunks=DASK_CHUNKS,
    )
    for var in ("extreme_events", "dat_anomaly", "thresholds", "mask", "dat_stn", "STD", "extreme_events_stn", "thresholds_stn"):
        assert var in ds.data_vars, f"missing {var}"
    assert ds.attrs["method_anomaly"] == "detrend_harmonic"
    assert ds.attrs["method_extreme"] == "global_percentile"
    assert ds.attrs["threshold_percentile"] == 95
    # Boolean flags are stored as int8 (0/1) so the dataset stays NetCDF-saveable.
    assert bool(ds.attrs["standardise"]) is True
    assert ds.extreme_events.dtype == bool and ds.extreme_events_stn.dtype == bool
    assert ds.dat_anomaly.dtype == np.float32 and ds.dat_stn.dtype == np.float32 and ds.STD.dtype == np.float32
    assert {"time", "lat", "lon"} <= set(ds.dat_stn.dims)
    assert {"dayofyear", "lat", "lon"} <= set(ds.STD.dims) and "time" not in ds.STD.dims
    assert_percentile_frequency(float(ds.extreme_events.mean()), 95, description="regular extreme events (standardise=True)")
    assert_percentile_frequency(
        float(ds.extreme_events_stn.mean()), 95, description="standardised extreme events (standardise=True)"
    )


def test_reference_period_wiring():
    """preprocess_data stores ``reference_period`` on the output, and rejects it for shifting_baseline."""
    times = pd.date_range("2000-01-01", "2001-12-31", freq="D")
    da = xr.DataArray(
        np.random.default_rng(0).standard_normal((len(times), 3, 3)).astype(np.float32),
        dims=["time", "lat", "lon"],
        coords={"time": times, "lat": [0.0, 1.0, 2.0], "lon": [0.0, 1.0, 2.0]},
    ).chunk({"time": -1})
    result = marEx.preprocess_data(
        da,
        method_anomaly="fixed_baseline",
        method_extreme="global_percentile",
        threshold_percentile=95,
        reference_period=(2000, 2000),
        dimensions=GRID_DIMS,
        dask_chunks={"time": -1},
    )
    assert result.attrs.get("reference_period") == [2000, 2000]

    with pytest.raises(ConfigurationError, match="reference_period is not supported"):
        marEx.preprocess_data(
            gridded_input(),
            method_anomaly="shifting_baseline",
            reference_period=(1990, 2010),
            dimensions=GRID_DIMS,
            dask_chunks=DASK_CHUNKS,
        )


def test_extreme_steps_wording():
    """Exact-string table: ``preprocessing_steps`` names the cadence it ran on and the tail it took.

    ``_extreme_steps`` once said "Day-of-year thresholds" on a monthly and a 6-hourly axis alike and
    never mentioned the tail. Its new parameters default to the old behaviour, so the daily
    upper-tail string every run has ever recorded is unchanged, character for character.
    ``window_days`` is physical, so the window half never changes units; and wording must never be
    the thing that raises: an unknown index name says what it always did.
    """
    rows = {
        "daily-upper-seasonal-3-args": (("seasonal_percentile", 11, None), ["Day-of-year thresholds with 11 day window"]),
        "daily-upper-seasonal-explicit": (
            ("seasonal_percentile", 11, None, "dayofyear", "upper"),
            ["Day-of-year thresholds with 11 day window"],
        ),
        "daily-upper-global-3-args": (("global_percentile", 11, None), ["Global percentile threshold applied to all days"]),
        "daily-upper-global-explicit": (
            ("global_percentile", 11, None, "dayofyear", "upper"),
            ["Global percentile threshold applied to all days"],
        ),
        "monthly": (("seasonal_percentile", 11, None, "month", "upper"), ["Monthly thresholds with 11 day window"]),
        "hour-of-year": (("seasonal_percentile", 11, None, "hourofyear", "upper"), ["Hour-of-year thresholds with 11 day window"]),
        "lower-seasonal": (
            ("seasonal_percentile", 11, None, "dayofyear", "lower"),
            ["Day-of-year lower-tail thresholds with 11 day window"],
        ),
        "lower-global": (
            ("global_percentile", 11, None, "dayofyear", "lower"),
            ["Global lower-tail percentile threshold applied to all days"],
        ),
        "cadence-tail-spatial-window": (
            ("seasonal_percentile", 11, 5, "month", "lower"),
            ["Monthly lower-tail thresholds with 11 day window & 5 spatial neighbours"],
        ),
        "global-off-the-daily-axis": (
            ("global_percentile", 11, None, "hourofyear", "upper"),
            ["Global percentile threshold applied to all steps"],
        ),
        "unknown-index-falls-back": (
            ("seasonal_percentile", 11, None, "not-a-cycle", "upper"),
            ["Day-of-year thresholds with 11 day window"],
        ),
    }
    failures = []
    for case, (args, expected) in rows.items():

        def row(args=args, expected=expected):
            got = _extreme_steps(*args)
            assert got == expected, f"{got} != {expected}"

        check(failures, case, row)
    assert_no_failures(failures)


def _preprocessing_steps(
    method_anomaly, method_extreme, standardise=False, detrend_orders=(1,), window_years=15, smooth_days=21, window_days=11
):
    """Both stages' step descriptions, composed in execution order as the chained pipeline does."""
    return _anomaly_steps(method_anomaly, standardise, list(detrend_orders), window_years, smooth_days, None) + _extreme_steps(
        method_extreme, window_days, None
    )


# MEASURED on 7b9b4e3 (pins/detect/pins_steps_chunks.out) at detrend_orders=[1], window_years=15,
# smooth_days=21, window_days=11.
_ANOMALY_STEPS = {
    "detrend_harmonic": ["Removed polynomial trend orders=[1] & seasonal cycle"],
    "shifting_baseline": ["Rolling climatology using 15 years", "Smoothed with 21-day window"],
    "detrend_fixed_baseline": [
        "Removed polynomial trend orders=[1]",
        "Daily climatology computed from detrended data",
        "Climatology smoothed with 21-day circular window",
    ],
    "fixed_baseline": ["Daily climatology computed from full time series", "Climatology smoothed with 21-day circular window"],
}
_EXTREME_STEPS = {
    "global_percentile": ["Global percentile threshold applied to all days"],
    "seasonal_percentile": ["Day-of-year thresholds with 11 day window"],
}


def test_composed_preprocessing_steps():
    """Exact-string table of the composed ``preprocessing_steps``: all 8 combinations, plus the orders,
    standardise and shifting-window wording."""
    rows = {f"{a}/{e}": ((a, e), _ANOMALY_STEPS[a] + _EXTREME_STEPS[e]) for a, e in ALL_COMBOS}
    rows.update(
        {
            "detrend_harmonic-orders-[1, 2]": (
                ("detrend_harmonic", "global_percentile", False, (1, 2)),
                ["Removed polynomial trend orders=[1, 2] & seasonal cycle", "Global percentile threshold applied to all days"],
            ),
            "detrend_harmonic-standardise": (
                ("detrend_harmonic", "global_percentile", True),
                [
                    "Removed polynomial trend orders=[1] & seasonal cycle",
                    "Normalised by 30-day rolling STD",
                    "Global percentile threshold applied to all days",
                ],
            ),
            "shifting_baseline-10y-15d-5d": (
                ("shifting_baseline", "seasonal_percentile", False, (1,), 10, 15, 5),
                ["Rolling climatology using 10 years", "Smoothed with 15-day window", "Day-of-year thresholds with 5 day window"],
            ),
            "detrend_fixed_baseline-orders-[1, 2]": (
                ("detrend_fixed_baseline", "seasonal_percentile", False, (1, 2)),
                [
                    "Removed polynomial trend orders=[1, 2]",
                    "Daily climatology computed from detrended data",
                    "Climatology smoothed with 21-day circular window",
                    "Day-of-year thresholds with 11 day window",
                ],
            ),
        }
    )
    failures = []
    for case, (args, expected) in rows.items():

        def row(args=args, expected=expected):
            got = _preprocessing_steps(*args)
            assert got == expected, f"{got} != {expected}"

        check(failures, case, row)
    assert_no_failures(failures)


def test_neighbour_axis_is_not_a_spatial_dim():
    """``finalise`` makes the FIELD's spatial dims whole -- and nothing else's.

    ``ds.dims`` on the output is the union over every variable, so deriving the spatial dims from
    it would also pick up ``neighbours``' own ``nv`` axis on the unstructured path. That axis is
    not a spatial dimension of the field and must not be rechunked. Neither golden covers it (both
    are gridded), so it is pinned directly.
    """
    from marEx.core.dimensions import extra_dims, resolve_dims

    sst = xr.open_zarr(str(DATA_DIR / "sst_unstructured.zarr"), chunks={}).to.isel(time=slice(0, 3 * 365))
    ncells = sst.sizes["ncells"]
    sst = sst.assign_coords(
        lat=xr.DataArray(np.linspace(-90, 90, ncells), dims=["ncells"]),
        lon=xr.DataArray(np.linspace(-180, 180, ncells), dims=["ncells"]),
    ).chunk({"time": 30, "ncells": -1})
    neighbours = xr.DataArray(np.zeros((3, ncells), dtype=np.int32), dims=("nv", "ncells"), coords={"nv": np.arange(3)})
    cell_areas = xr.DataArray(np.ones(ncells, dtype=np.float32), dims=("ncells",))

    result = marEx.preprocess_data(
        sst,
        method_anomaly="fixed_baseline",
        method_extreme="global_percentile",
        threshold_percentile=95,
        dimensions=MESH_DIMS,
        coordinates=MESH_COORDS,
        neighbours=neighbours,
        cell_areas=cell_areas,
        dask_chunks={"time": 30},
    )

    # The field has no extra dims, so finalise must chunk exactly ncells and time.
    assert resolve_dims(sst, MESH_DIMS, MESH_COORDS).extra == ()
    # `nv` is visible on the output Dataset but is not the field's spatial dim.
    assert "nv" in result.dims
    assert "nv" in extra_dims(result, MESH_DIMS, exclude=("dayofyear",))
    assert result.neighbours.chunksizes["nv"] == (3,)
    assert result.dat_anomaly.chunksizes["ncells"] == (ncells,)
