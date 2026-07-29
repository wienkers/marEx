"""
Error handling: what each entry point rejects, with which exception type and message.

Every rejection is a row of a table, run through ``check`` so a failing row names its case and
the other rows still run. The inputs are tiny synthetic fields (3x4 cells); the two tests that
need a real time axis use a three-year daily series, which still computes in well under a second.
All tests run on the threaded scheduler.

Also covers the exception classes themselves: hierarchy, error codes, the formatted message
and the builtin-to-marEx wrapper.
"""

import re

import numpy as np
import pandas as pd
import pytest
import xarray as xr

import marEx
from marEx.exceptions import (
    ConfigurationError,
    CoordinateError,
    DataValidationError,
    DependencyError,
    MarExError,
    ProcessingError,
    TrackingError,
    VisualisationError,
    create_coordinate_error,
    create_data_validation_error,
    create_processing_error,
    wrap_exception,
)

from .conftest import assert_no_failures, check

DIMS = {"time": "time", "x": "lon", "y": "lat"}
CHUNKS = {"time": 25}


# ---------------------------------------------------------------------------------------------
# Helpers and synthetic inputs
# ---------------------------------------------------------------------------------------------
def rejects(call, exc, *patterns):
    """``call()`` must raise ``exc`` (or a subclass), and every regex in ``patterns`` must occur in the message.

    ``exc=None`` is a control row: the call must complete without raising.
    """
    kind = text = None
    try:
        call()
    except Exception as e:  # noqa: BLE001 - the type is asserted below, once the exception is released
        kind, text = type(e), str(e)
    if exc is None:
        assert kind is None, f"expected no error, got {kind.__name__}: {text.splitlines()[0] if text else ''}"
        return
    if kind is None:
        pytest.fail(f"expected {exc.__name__}, but nothing was raised")
    first_line = text.splitlines()[0] if text else ""
    assert issubclass(kind, exc), f"expected {exc.__name__}, got {kind.__name__}: {first_line}"
    for pattern in patterns:
        assert re.search(pattern, text), f"message lacks {pattern!r}: {first_line}"


def run_table(rows):
    """Run ``(case, exception, patterns, call)`` rows; all rows run and every failing case is reported."""
    failures = []
    for case, exc, patterns, call in rows:
        check(failures, case, lambda call=call, exc=exc, patterns=patterns: rejects(call, exc, *patterns))
    assert_no_failures(failures)


def daily(n_days=100, sd=1.0, seed=0, base=15.0, nlat=3, nlon=4):
    """A numpy-backed 3x4 daily field starting 2000-01-01: ``base`` plus white noise of std ``sd``."""
    rng = np.random.default_rng(seed)
    return xr.DataArray(
        rng.standard_normal((n_days, nlat, nlon)) * sd + base,
        dims=["time", "lat", "lon"],
        coords={
            "time": pd.date_range("2000-01-01", periods=n_days, freq="D"),
            "lat": np.linspace(-10, 10, nlat),
            "lon": np.linspace(0, 20, nlon),
        },
    )


def lazy(da, chunks=CHUNKS):
    return da.chunk(chunks)


def events_field(lat, lon, n_time=10, seed=0):
    """A chunked boolean field on the given lat/lon axes and a chunked boolean mask of ones."""
    rng = np.random.default_rng(seed)
    data = xr.DataArray(
        rng.random((n_time, len(lat), len(lon))) > 0.6,
        dims=["time", "lat", "lon"],
        coords={"time": np.arange(n_time), "lat": lat, "lon": lon},
    ).chunk({"time": 5, "lat": -1, "lon": -1})
    mask = xr.ones_like(data.isel(time=0), dtype=bool).chunk({"lat": -1, "lon": -1})
    return data, mask


GLOBAL_AXES = (np.linspace(-90, 90, 3), np.linspace(-180, 180, 4))
REGIONAL_AXES = (np.linspace(0, 5, 3), np.linspace(0, 10, 4))
TRACK = {"R_fill": 1, "area_filter_quartile": 0.5}


def mesh_events(n_cells=20, n_time=6, lat=True, lon=True, seed=1):
    """A chunked boolean field on an unstructured ``cell`` dimension and an all-True mask."""
    rng = np.random.default_rng(seed)
    coords = {"time": range(n_time)}
    if lat:
        coords["lat"] = ("cell", rng.uniform(-90, 90, n_cells))
    if lon:
        coords["lon"] = ("cell", rng.uniform(-180, 180, n_cells))
    data = xr.DataArray(rng.random((n_time, n_cells)) > 0.5, dims=["time", "cell"], coords=coords).chunk({"time": 3})
    return data, xr.ones_like(data.isel(time=0), dtype=bool)


# ---------------------------------------------------------------------------------------------
# The three stages
# ---------------------------------------------------------------------------------------------
def test_anomaly_stage_rejects_bad_input():
    """compute_normalised_anomaly names what is wrong: a numpy array, an unknown method, missing dimensions or coordinates."""
    field = lazy(daily())
    in_memory = daily()
    missing = {"time": "time", "x": "missing_x", "y": "missing_y"}
    small = lazy(daily(10, nlat=5, nlon=5), {"time": 5})

    def anomaly(da, **kwargs):
        return lambda: marEx.anomaly.compute_normalised_anomaly(da, method_anomaly="detrend_harmonic", **kwargs)

    run_table(
        [
            ("numpy-backed input", DataValidationError, [r"must be Dask-backed"], anomaly(in_memory, dimensions=DIMS)),
            (
                "unknown method",
                ConfigurationError,
                [r"Unknown anomaly method 'invalid_method'"],
                lambda: marEx.anomaly.compute_normalised_anomaly(field, method_anomaly="invalid_method", dimensions=DIMS),
            ),
            (
                "unknown method, default dimensions",
                ConfigurationError,
                [r"Unknown anomaly method 'invalid_method'"],
                lambda: marEx.anomaly.compute_normalised_anomaly(small, method_anomaly="invalid_method"),
            ),
            ("missing dimensions", DataValidationError, [r"Missing required dimensions"], anomaly(field, dimensions=missing)),
            (
                "missing coordinates",
                DataValidationError,
                [r"Missing required coordinates"],
                anomaly(field, dimensions=DIMS, coordinates=missing),
            ),
        ]
    )


def test_extremes_stage_rejects_bad_configuration():
    """identify_extremes rejects unknown methods, incompatible options, even windows and missing dimensions or coordinates."""
    field = lazy(daily())
    small = lazy(daily(10, nlat=5, nlon=5), {"time": 5})
    missing = {"time": "time", "x": "missing_x", "y": "missing_y"}
    grid = {"dimensions": DIMS, "coordinates": DIMS}

    def identify(da=field, **kwargs):
        return lambda: marEx.extremes.identify_extremes(da, **kwargs)

    run_table(
        [
            (
                "unknown method_percentile",
                ConfigurationError,
                [r"Unknown method_percentile 'invalid_method'"],
                identify(method_percentile="invalid_method"),
            ),
            (
                "precision with exact",
                ConfigurationError,
                [r"Parameter 'precision' cannot be used with method_percentile='exact'"],
                identify(method_percentile="exact", precision=0.02),
            ),
            (
                "max_anomaly with exact",
                ConfigurationError,
                [r"Parameter 'max_anomaly' cannot be used with method_percentile='exact'"],
                identify(method_percentile="exact", max_anomaly=10.0),
            ),
            (
                "window_spatial with global_percentile",
                ConfigurationError,
                [r"window_spatial can only be used with method_extreme='seasonal_percentile'"],
                identify(method_extreme="global_percentile", window_spatial=3, **grid),
            ),
            (
                "window_spatial with exact",
                ConfigurationError,
                [r"window_spatial is not supported with method_percentile='exact'"],
                identify(method_extreme="seasonal_percentile", method_percentile="exact", window_spatial=3, **grid),
            ),
            (
                "even window_days",
                ConfigurationError,
                [r"window_days must be an odd number"],
                identify(method_extreme="seasonal_percentile", window_days=10, **grid),
            ),
            (
                "even window_spatial",
                ConfigurationError,
                [r"window_spatial must be an odd number"],
                identify(method_extreme="seasonal_percentile", method_percentile="approximate", window_spatial=4, **grid),
            ),
            (
                "unknown method_extreme",
                ConfigurationError,
                [r"Unknown extreme method 'invalid_extreme'"],
                identify(method_extreme="invalid_extreme", **grid),
            ),
            (
                "unknown method_extreme, default dimensions",
                ConfigurationError,
                [r"Unknown extreme method 'invalid_method'"],
                identify(small, method_extreme="invalid_method"),
            ),
            (
                "missing dimensions",
                DataValidationError,
                [r"Missing required dimensions"],
                identify(method_extreme="global_percentile", dimensions=missing),
            ),
            (
                "missing coordinates",
                DataValidationError,
                [r"Missing required coordinates"],
                identify(method_extreme="global_percentile", dimensions=DIMS, coordinates=missing),
            ),
        ]
    )


def test_preprocess_rejects_bad_structure():
    """preprocess_data rejects a numpy array, wrong or missing dimensions and coordinates, and bad options, before computing."""
    field = lazy(daily())
    in_memory = daily()
    small = daily(10, nlat=5, nlon=5)
    small_in_memory = small
    wrong_names = {"time": "time", "x": "wrong_lon", "y": "wrong_lat"}
    no_such_axes = {"time": "nonexistent_time", "x": "nonexistent_x", "y": "nonexistent_y"}
    no_such_coords = {"time": "nonexistent_time_coord", "x": "nonexistent_x_coord", "y": "nonexistent_y_coord"}
    stacked = lazy(daily().stack(ncells=("lat", "lon")))
    mesh_dims = {"time": "time", "x": "ncells"}
    mesh_coords = {"time": "time", "x": "ncells", "y": "ncells"}

    def preprocess(da=field, **kwargs):
        return lambda: marEx.preprocess_data(da, dask_chunks=CHUNKS, **kwargs)

    run_table(
        [
            (
                "numpy-backed input names Dask and chunking",
                DataValidationError,
                [r"Input DataArray must be Dask-backed", r"\.chunk\("],
                preprocess(in_memory, dimensions=DIMS),
            ),
            (
                "numpy-backed input, default dimensions",
                DataValidationError,
                [r"Input DataArray must be Dask-backed"],
                lambda: marEx.preprocess_data(small_in_memory),
            ),
            ("no x entry", DataValidationError, [r"missing the required 'x' entry"], preprocess(dimensions={"y": "lat"})),
            (
                "dimension names not in the data",
                DataValidationError,
                [r"Missing required dimensions"],
                preprocess(dimensions=wrong_names),
            ),
            (
                "no dimension named like any in the data",
                DataValidationError,
                [r"Missing required dimensions"],
                preprocess(dimensions=no_such_axes),
            ),
            (
                "coordinate names not in the data",
                DataValidationError,
                [r"Missing required coordinates"],
                preprocess(dimensions=DIMS, coordinates=no_such_coords),
            ),
            (
                "unstructured without coordinates",
                DataValidationError,
                [r"Coordinates parameter must be explicitly specified for unstructured data"],
                preprocess(dimensions={"time": "time", "x": "lon"}, coordinates=None),
            ),
            (
                "detrend_orders empty",
                ConfigurationError,
                [r"detrend_orders cannot be empty"],
                preprocess(method_anomaly="detrend_fixed_baseline", detrend_orders=[], dimensions=DIMS),
            ),
            (
                "detrend_orders negative",
                ConfigurationError,
                [r"Invalid polynomial orders: \[-1\]"],
                preprocess(method_anomaly="detrend_fixed_baseline", detrend_orders=[-1], dimensions=DIMS),
            ),
            (
                # The coordinates here map x and y onto one name, which must pass validation: it is the
                # unstructured success path up to the point the extremes stage rejects the option.
                "window_spatial on an unstructured grid",
                ConfigurationError,
                [r"window_spatial is not supported for unstructured grids"],
                preprocess(
                    stacked,
                    dimensions=mesh_dims,
                    coordinates=mesh_coords,
                    method_anomaly="detrend_harmonic",
                    method_extreme="seasonal_percentile",
                    method_percentile="approximate",
                    window_spatial=3,
                    window_days=5,
                ),
            ),
        ]
    )


def test_coordinates_without_time_gain_the_time_dimension():
    """Coordinates given without a ``time`` entry get one, named after the time dimension, ahead of the others."""
    from marEx.core.validation import _infer_dims_coords

    da = xr.DataArray(
        np.zeros((3, 5), dtype=np.float32),
        dims=("t", "ncells"),
        coords={"t": np.arange(3), "lon": ("ncells", np.linspace(0, 40, 5)), "lat": ("ncells", np.zeros(5))},
    )
    dimensions, coordinates = _infer_dims_coords(da, {"time": "t", "x": "ncells"}, {"x": "lon", "y": "lat"})
    assert dimensions == {"time": "t", "x": "ncells"}, dimensions
    assert list(coordinates.items()) == [("time", "t"), ("x", "lon"), ("y", "lat")], coordinates


def test_preprocess_rejects_non_finite_data():
    """A field with no finite values, or with invalid values at some locations, is rejected with a count of them."""
    all_nan = daily()
    all_nan.values[:] = np.nan
    all_inf = daily()
    all_inf.values[:] = np.inf
    partial = daily(sd=5.0)
    partial.values[:, 1, 2] = np.nan  # one location all NaN
    partial.values[50:, 0, 1] = np.inf  # another partly infinite
    partial.values[:20, 2, 0] = np.nan  # another partly NaN

    def preprocess(da):
        return lambda: marEx.preprocess_data(lazy(da), dimensions=DIMS, dask_chunks=CHUNKS, method_anomaly="detrend_harmonic")

    run_table(
        [
            ("all NaN", DataValidationError, [r"contains no valid.*finite.*data"], preprocess(all_nan)),
            ("all infinite", DataValidationError, [r"contains no valid.*finite.*data"], preprocess(all_inf)),
            (
                "invalid values at some locations",
                DataValidationError,
                [r"contains.*invalid values at.*locations"],
                preprocess(partial),
            ),
        ]
    )


def test_shifting_baseline_needs_window_years_of_data():
    """Fewer years than window_years raises DataValidationError naming both year counts, never a bare IndexError."""
    rows = []
    for n_years, window_years in [(2, 15), (3, 5), (2, 50)]:
        field = lazy(daily(365 * n_years))
        rows.append(
            (
                f"{n_years} years of data, window_years={window_years}",
                DataValidationError,
                [
                    r"Insufficient data for shifting_baseline method",
                    rf"spans {n_years} years",
                    rf"requires at least {window_years} years",
                    rf"Reduce window_years parameter \(currently {window_years}\)",
                ],
                lambda field=field, window_years=window_years: marEx.preprocess_data(
                    field,
                    method_anomaly="shifting_baseline",
                    method_extreme="seasonal_percentile",
                    window_years=window_years,
                    dimensions=DIMS,
                    dask_chunks=CHUNKS,
                ),
            )
        )
    run_table(rows)


def test_threshold_beyond_max_anomaly_raises():
    """A threshold that saturates the top histogram bin is an error naming the range, on both extreme methods."""
    field = daily(200, sd=20.0)
    field.values[:50, 0, 0] += 50  # large positive anomalies push the high quantiles out of range

    def preprocess(method_extreme, **kwargs):
        return lambda: marEx.preprocess_data(
            lazy(field),
            dimensions=DIMS,
            dask_chunks=CHUNKS,
            method_anomaly="detrend_harmonic",
            method_extreme=method_extreme,
            method_percentile="approximate",
            threshold_percentile=99,
            max_anomaly=5.0,
            **kwargs,
        )

    patterns = [r"exceed expected range"]
    run_table(
        [
            ("global_percentile", ConfigurationError, patterns, preprocess("global_percentile")),
            ("seasonal_percentile", ConfigurationError, patterns, preprocess("seasonal_percentile", window_days=5)),
        ]
    )


def test_degenerate_variance_cells_complete():
    """Cells with no or almost no variance compute: a constant cell has no events and the other cells are unaffected."""
    n_days = 1096  # three years
    constant = daily(n_days, sd=2.0)
    constant.values[:, 0, 0] = 15.0
    near_constant = daily(n_days, sd=0.01, base=10.0, seed=3)
    near_constant.values[:, 0, 0] = 10.0 + np.random.default_rng(4).standard_normal(n_days) * 1e-4
    chunks = {"time": 100}

    def events_per_cell(field, method_extreme, threshold_percentile, **kwargs):
        ds = marEx.preprocess_data(
            lazy(field, chunks),
            dimensions=DIMS,
            dask_chunks=chunks,
            method_anomaly="detrend_harmonic",
            method_extreme=method_extreme,
            threshold_percentile=threshold_percentile,
            **kwargs,
        )
        assert isinstance(ds, xr.Dataset), f"preprocess_data returned {type(ds).__name__}"
        counts = ds.extreme_events.compute().sum("time").values
        anomaly = ds.dat_anomaly.compute().isel(lat=0, lon=0).values
        del ds
        return counts, anomaly

    def constant_cell_has_no_events(method_extreme, **kwargs):
        counts, anomaly = events_per_cell(constant, method_extreme, 95, **kwargs)
        assert np.isfinite(anomaly).all() and np.abs(anomaly).max() < 1e-4, "the constant cell's anomaly is not zero"
        assert counts[0, 0] == 0, f"the constant cell has {counts[0, 0]} events, expected 0"

    def near_constant_cell_is_flagged_like_the_rest():
        counts, _ = events_per_cell(near_constant, "global_percentile", 99)
        # 1 % of 1096 days is 11: a tiny but non-zero variance still gives the cell its share of events.
        assert counts.min() >= 6 and counts.max() <= 16, f"events per cell {counts.ravel().tolist()}, expected about 11 each"

    failures = []
    # On the exact path a constant cell's threshold equals the constant, so every tie would flag unless clamped.
    # (The exact global path is left out: its first call in a process costs about 13 s of compilation.)
    for method_extreme, kwargs in [
        ("global_percentile", {}),
        ("seasonal_percentile", {"window_days": 5}),
        ("seasonal_percentile", {"window_days": 5, "method_percentile": "exact"}),
    ]:
        case = f"constant cell, {method_extreme} {kwargs.get('method_percentile', 'approximate')}"
        check(failures, case, lambda m=method_extreme, k=kwargs: constant_cell_has_no_events(m, **k))
    check(failures, "near-constant cell, global_percentile 99", near_constant_cell_is_flagged_like_the_rest)
    assert_no_failures(failures)


# ---------------------------------------------------------------------------------------------
# The tracker
# ---------------------------------------------------------------------------------------------
def test_tracker_rejects_bad_coordinates():
    """tracker names a bad coordinate_units value or a range it cannot read as degrees or radians."""
    regional, regional_mask = events_field(*REGIONAL_AXES)
    stacked = events_field(*GLOBAL_AXES)[0].stack(ncells=("lat", "lon")).chunk({"time": 5, "ncells": -1})
    stacked_mask = xr.ones_like(stacked.isel(time=0), dtype=bool)

    run_table(
        [
            (
                "regional mode without coordinate_units",
                CoordinateError,
                [r"coordinate_units must be specified when regional_mode=True"],
                lambda: marEx.tracker(regional, regional_mask, regional_mode=True, coordinate_units=None, **TRACK),
            ),
            (
                "regional mode, invalid coordinate_units",
                CoordinateError,
                [r"Invalid coordinate_units 'invalid_units'"],
                lambda: marEx.tracker(regional, regional_mask, regional_mode=True, coordinate_units="invalid_units", **TRACK),
            ),
            (
                "global mode, invalid coordinate_units",
                CoordinateError,
                [r"Invalid coordinate_units 'invalid_units'"],
                lambda: marEx.tracker(regional, regional_mask, regional_mode=False, coordinate_units="invalid_units", **TRACK),
            ),
            (
                "global mode, range too small to be degrees or radians",
                CoordinateError,
                [r"Cannot auto-detect coordinate units from range", r"(?i)coordinate", r"(?i)range"],
                lambda: marEx.tracker(regional, regional_mask, regional_mode=False, coordinate_units=None, **TRACK),
            ),
            (
                "unstructured, x and y mapped onto the time coordinate",
                CoordinateError,
                [r"Cannot auto-detect coordinate units from range"],
                lambda: marEx.tracker(
                    stacked,
                    stacked_mask,
                    dimensions={"time": "time", "x": "ncells"},
                    coordinates={"time": "time", "x": "time", "y": "time"},
                    regional_mode=False,
                    **TRACK,
                ),
            ),
        ]
    )


def test_tracker_rejects_bad_data_and_parameters():
    """tracker rejects unchunked data, non-boolean data or mask, an all-False mask, bad parameters and wrong dimensions."""
    data, mask = events_field(*GLOBAL_AXES)
    regional, regional_mask = events_field(*REGIONAL_AXES)
    run_table(
        [
            (
                "unchunked data and mask, global coordinates",
                DataValidationError,
                [r"Data must be chunked"],
                lambda: marEx.tracker(data.compute(), mask.compute(), **TRACK),
            ),
            (
                "unchunked data, chunked mask",
                DataValidationError,
                [r"Data must be chunked"],
                lambda: marEx.tracker(data.compute(), mask, **TRACK),
            ),
            (
                "unchunked data, regional mode",
                DataValidationError,
                [r"Data must be chunked"],
                lambda: marEx.tracker(
                    regional.compute(), regional_mask.compute(), regional_mode=True, coordinate_units="degrees", **TRACK
                ),
            ),
            (
                "float data",
                DataValidationError,
                [r"Input DataArray must be binary \(boolean type\)"],
                lambda: marEx.tracker(data.astype(float), mask, **TRACK),
            ),
            (
                "float mask",
                DataValidationError,
                [r"Mask must be binary \(boolean type\)"],
                lambda: marEx.tracker(data, mask.astype(float), **TRACK),
            ),
            (
                "all-False mask",
                DataValidationError,
                [r"Mask contains only False values"],
                lambda: marEx.tracker(data, xr.zeros_like(mask, dtype=bool), **TRACK),
            ),
            (
                "area_filter_quartile below 0",
                ConfigurationError,
                [r"Invalid area_filter_quartile value"],
                lambda: marEx.tracker(data, mask, R_fill=1, area_filter_quartile=-0.1),
            ),
            (
                # The same call with a valid value must construct, or every row could pass by rejecting everything.
                "area_filter_quartile 0.5 is accepted",
                None,
                [],
                lambda: marEx.tracker(data, mask, **TRACK),
            ),
            (
                "area_filter_quartile above 1",
                ConfigurationError,
                [r"Invalid area_filter_quartile value"],
                lambda: marEx.tracker(data, mask, R_fill=1, area_filter_quartile=1.5),
            ),
            (
                "odd T_fill",
                ConfigurationError,
                [r"T_fill must be even for temporal symmetry"],
                lambda: marEx.tracker(data, mask, T_fill=3, **TRACK),
            ),
            (
                "dimension names not in the data",
                DataValidationError,
                [r"Invalid dimensions for gridded data"],
                lambda: marEx.tracker(
                    data,
                    mask,
                    dimensions={"time": "time", "x": "nonexistent_x", "y": "nonexistent_y"},
                    coordinates={"time": "time", "x": "lon", "y": "lat"},
                    **TRACK,
                ),
            ),
        ]
    )


def test_unstructured_tracker_requires_grid_inputs(tmp_path):
    """An unstructured tracker names a missing temp_dir, neighbours, cell_areas or coordinate, and a wrong neighbour array."""
    dims = {"time": "time", "x": "cell"}
    coords = {"time": "time", "x": "lon", "y": "lat"}
    base = {"unstructured_grid": True, "coordinate_units": "degrees", "dimensions": dims, "coordinates": coords, **TRACK}
    cells = 20
    areas = xr.DataArray(np.ones(cells), dims=["cell"], coords={"cell": range(cells)})
    four_rows = xr.DataArray(np.ones((4, cells)), dims=["nv", "cell"], coords={"cell": range(cells), "nv": range(4)})
    wrong_names = xr.DataArray(
        np.ones((3, cells)), dims=["wrong_nv", "wrong_cell"], coords={"wrong_cell": range(cells), "wrong_nv": range(3)}
    )
    data, mask = mesh_events(cells)
    no_coords, no_coords_mask = mesh_events(cells, lat=False, lon=False)
    no_lat, no_lat_mask = mesh_events(cells, lat=False)

    rows = [
        (
            "temp_dir missing",
            [r"temp_dir is required for unstructured grids"],
            lambda: marEx.tracker(data, mask, temp_dir=None, **base),
        ),
        (
            "neighbours missing",
            [r"neighbours array is required for unstructured grids"],
            lambda: marEx.tracker(data, mask, temp_dir=str(tmp_path), neighbours=None, **base),
        ),
        (
            "cell_areas missing",
            [r"cell_areas array is required for unstructured grids"],
            lambda: marEx.tracker(data, mask, temp_dir=str(tmp_path), neighbours=np.ones((3, cells)), cell_areas=None, **base),
        ),
        (
            "dimensions not in the data",
            [r"Invalid dimensions for unstructured data"],
            lambda: marEx.tracker(data, mask, **{**base, "dimensions": {"time": "time", "x": "invalid_dim"}}),
        ),
        (
            "neighbours with four rows",
            [r"Invalid neighbour array for triangular grid"],
            lambda: marEx.tracker(data, mask, temp_dir=str(tmp_path), neighbours=four_rows, cell_areas=areas, **base),
        ),
        (
            "neighbours with the wrong dimension names",
            [r"Invalid neighbour array dimensions"],
            lambda: marEx.tracker(data, mask, temp_dir=str(tmp_path), neighbours=wrong_names, cell_areas=areas, **base),
        ),
        (
            "no coordinates in the data",
            [r"Missing required coordinates"],
            lambda: marEx.tracker(
                no_coords,
                no_coords_mask,
                dimensions=dims,
                coordinates={"time": "time", "x": "missing_lon", "y": "missing_lat"},
                **TRACK,
            ),
        ),
        (
            # Only y is absent: the friendly error, not a bare KeyError from capturing the coordinates early.
            "only the y coordinate missing",
            [r"Missing required coordinates"],
            lambda: marEx.tracker(
                no_lat,
                no_lat_mask,
                dimensions=dims,
                coordinates=coords,
                unstructured_grid=True,
                coordinate_units="degrees",
                **TRACK,
            ),
        ),
    ]

    run_table([(case, DataValidationError, patterns, call) for case, patterns, call in rows])


def test_wrap_exception_maps_builtin_errors():
    """wrap_exception turns builtin errors into the matching marEx class, keeping the original as __cause__."""
    failures = []

    def wraps(original, expected, message, *args, **kwargs):
        wrapped = wrap_exception(original, *args, **kwargs)
        assert isinstance(wrapped, expected), f"wrapped as {type(wrapped).__name__}, expected {expected.__name__}"
        assert isinstance(wrapped, MarExError), "the wrapper is not a MarExError"
        if message is not None:
            assert wrapped.message == message, f"message is {wrapped.message!r}, expected {message!r}"
        assert wrapped.__cause__ is original, "the original error is not the __cause__"
        return wrapped

    def value_error():
        wrapped = wraps(ValueError("Original error message"), DataValidationError, "Original error message")
        assert "Original ValueError" in wrapped.details, f"details are {wrapped.details!r}"

    def chained():
        try:
            try:
                raise ValueError("Original error")
            except ValueError as e:
                raise DataValidationError("Validation failed", details="Caused by value error") from e
        except DataValidationError as caught:
            assert isinstance(caught.__cause__, ValueError), f"cause is {caught.__cause__!r}"
            assert str(caught.__cause__) == "Original error", f"cause message is {str(caught.__cause__)!r}"

    check(failures, "ValueError -> DataValidationError", value_error)
    check(
        failures, "RuntimeError -> ProcessingError", lambda: wraps(RuntimeError("Runtime issue"), ProcessingError, "Runtime issue")
    )
    check(
        failures,
        "KeyError with a custom message -> ConfigurationError",
        lambda: wraps(KeyError("missing_key"), ConfigurationError, "Custom message", "Custom message"),
    )
    check(
        failures,
        "ImportError with a custom type -> DependencyError",
        lambda: wraps(ImportError("module not found"), DependencyError, None, marex_exception_type=DependencyError),
    )
    check(failures, "raise ... from keeps the cause", chained)
    assert_no_failures(failures)


# ---------------------------------------------------------------------------------------------
# The exception classes
# ---------------------------------------------------------------------------------------------
def test_exception_hierarchy_and_codes():
    """Every class is a MarExError and an Exception with its own error code; the convenience constructors fill the context."""
    failures = []
    codes = {
        DataValidationError: "DATA_VALIDATION",
        CoordinateError: "COORDINATE_ERROR",
        ProcessingError: "PROCESSING_ERROR",
        ConfigurationError: "CONFIGURATION_ERROR",
        DependencyError: "DEPENDENCY_ERROR",
        TrackingError: "TRACKING_ERROR",
        VisualisationError: "VISUALISATION_ERROR",
    }

    def class_row(cls, code):
        error = cls("something")
        assert isinstance(error, MarExError), f"{cls.__name__} is not a MarExError"
        assert isinstance(error, Exception), f"{cls.__name__} is not an Exception"
        assert error.error_code == code, f"error_code is {error.error_code!r}, expected {code!r}"

    for cls, code in codes.items():
        check(failures, f"class {cls.__name__}", lambda cls=cls, code=code: class_row(cls, code))

    def base_class():
        assert isinstance(MarExError("test"), Exception), "MarExError is not an Exception"

    def data_validation_constructor():
        info = {"dtype": "float32", "shape": (100, 200)}
        error = create_data_validation_error("Invalid data type", data_info=info, details="Expected boolean array")
        assert isinstance(error, DataValidationError), f"built a {type(error).__name__}"
        assert error.message == "Invalid data type", f"message is {error.message!r}"
        assert error.details == "Expected boolean array", f"details are {error.details!r}"
        assert error.context["dtype"] == "float32", f"context is {error.context}"
        assert error.context["shape"] == (100, 200), f"context is {error.context}"

    def coordinate_constructor():
        ranges = {"lat": (-90, 90), "lon": (0, 360)}
        error = create_coordinate_error("Invalid coordinate ranges", coordinate_ranges=ranges, detected_system="degrees")
        assert isinstance(error, CoordinateError), f"built a {type(error).__name__}"
        assert error.context["coordinate_ranges"] == ranges, f"context is {error.context}"
        assert error.context["detected_system"] == "degrees", f"context is {error.context}"

    def processing_constructor():
        error = create_processing_error("Out of memory", computation_info={"memory_gb": 16, "chunk_size": "1GB"})
        assert isinstance(error, ProcessingError), f"built a {type(error).__name__}"
        assert error.context["memory_gb"] == 16, f"context is {error.context}"
        assert error.context["chunk_size"] == "1GB", f"context is {error.context}"

    check(failures, "MarExError base class", base_class)
    check(failures, "create_data_validation_error", data_validation_constructor)
    check(failures, "create_coordinate_error", coordinate_constructor)
    check(failures, "create_processing_error", processing_constructor)
    assert_no_failures(failures)


def test_error_message_formatting():
    """The formatted message carries details, context, suggestions and the code; absent sections stay absent; add_* shows up."""
    failures = []

    def bare():
        error = MarExError("Something went wrong")
        assert str(error) == "Something went wrong", f"str is {str(error)!r}"
        assert error.message == "Something went wrong"
        assert error.details is None
        assert error.suggestions == []
        assert error.error_code is None
        assert error.context == {}

    def full():
        error = MarExError(
            message="Data validation failed",
            details="Array shape mismatch",
            suggestions=["Check input dimensions", "Verify data format"],
            error_code="DATA_001",
            context={"expected_shape": (100, 200), "actual_shape": (100, 150)},
        )
        assert error.message == "Data validation failed"
        assert error.details == "Array shape mismatch"
        assert error.suggestions == ["Check input dimensions", "Verify data format"]
        assert error.error_code == "DATA_001"
        assert error.context["expected_shape"] == (100, 200)

    def formatted():
        error = MarExError(
            message="Test error",
            details="Additional information",
            suggestions=["Try this", "Or this"],
            error_code="TEST_001",
            context={"param": "value"},
        )
        text = str(error)
        for fragment in (
            "Test error",
            "Details: Additional information",
            "Context: param=value",
            "Suggestions:",
            "- Try this",
            "- Or this",
            "Error Code: TEST_001",
        ):
            assert fragment in text, f"{fragment!r} is missing from {text!r}"

    def empty_sections_absent():
        text = str(VisualisationError("Plot failed"))
        for heading in ("Details:", "Context:", "Suggestions:"):
            assert heading not in text, f"{heading!r} appears although nothing was given: {text!r}"

    def add_after_creation():
        error = MarExError("Test error")
        assert len(error.suggestions) == 0 and len(error.context) == 0
        error.add_suggestion("New suggestion")
        error.add_context("key", "value")
        assert error.suggestions == ["New suggestion"], f"suggestions are {error.suggestions}"
        assert error.context["key"] == "value", f"context is {error.context}"

    def suggestions_are_actionable():
        error = DataValidationError(
            "Data must be Dask-backed",
            suggestions=[
                "Convert to Dask: data.chunk({'time': 30})",
                "Load with chunking: xr.open_dataset('file.nc').chunk()",
            ],
        )
        text = str(error)
        for fragment in ("Convert to Dask", "chunk({'time': 30})", "xr.open_dataset"):
            assert fragment in text, f"{fragment!r} is missing from {text!r}"

    def context_gives_debugging_info():
        error = ProcessingError(
            "Memory limit exceeded",
            context={"required_memory_gb": 32, "available_memory_gb": 16, "data_shape": (1000, 2000, 3000)},
        )
        text = str(error)
        for fragment in ("required_memory_gb=32", "available_memory_gb=16", "data_shape=(1000, 2000, 3000)"):
            assert fragment in text, f"{fragment!r} is missing from {text!r}"

    def attributes_survive_subclassing():
        error = TrackingError(
            "Tracking failed",
            details="Object merge conflict",
            suggestions=["Increase overlap threshold"],
            context={"n_objects": 1000, "merge_limit": 500},
        )
        assert error.message == "Tracking failed"
        assert error.details == "Object merge conflict"
        assert len(error.suggestions) == 1
        assert error.context["n_objects"] == 1000
        assert error.error_code == "TRACKING_ERROR"

    check(failures, "message only", bare)
    check(failures, "all parameters stored", full)
    check(failures, "every section formatted", formatted)
    check(failures, "empty sections absent", empty_sections_absent)
    check(failures, "add_suggestion and add_context", add_after_creation)
    check(failures, "suggestions are actionable", suggestions_are_actionable)
    check(failures, "context gives debugging info", context_gives_debugging_info)
    check(failures, "attributes survive subclassing", attributes_survive_subclassing)
    assert_no_failures(failures)
