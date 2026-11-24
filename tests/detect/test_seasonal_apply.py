"""Exact parity and bounded task sizes for the seasonal gather."""

import operator
from pathlib import Path

import dask.array as dsa
import numpy as np
import pytest
import xarray as xr

import marEx
from marEx.anomaly import fixed_baseline
from marEx.core import seasonal_apply
from marEx.core.time_axis import SeasonalCycle, infer_cycle
from marEx.extremes import seasonal_percentile

_OPERATIONS = {"subtract": operator.sub, "divide": operator.truediv, "upper": operator.ge, "lower": operator.le}


def _legacy_apply(da, lookup, values, *, timedim, cycledim, operation):
    # Use positional indexes to restore the rows dropped by UniqueGrouper, including
    # NaT rows and repeated timestamps. The validity rule applies before grouping.
    positions = np.arange(da.sizes[timedim])
    labels = lookup[cycledim].values
    valid = (values > 0) & np.isin(values, labels)
    indexed = da.drop_vars(cycledim, errors="ignore").assign_coords({timedim: positions, cycledim: (timedim, values)})
    if not valid.all():
        indexed = indexed.isel({timedim: valid})
    grouped = indexed.groupby({cycledim: xr.groupers.UniqueGrouper(labels=labels)})
    result = _OPERATIONS[operation](grouped, lookup).drop_vars(cycledim)
    if not valid.all():
        result = result.reindex({timedim: positions}, fill_value=np.nan if operation in ("subtract", "divide") else False)
    coords = {name: coord for name, coord in da.coords.items() if name != cycledim}
    result = result.transpose(*da.dims)
    if da.chunks is not None:
        result = result.chunk(dict(zip(da.dims, da.chunks)))
    return xr.DataArray(result.data, dims=da.dims, coords=coords, name=da.name, attrs=result.attrs)


def _assert_bits(actual, expected):
    assert actual.dtype == expected.dtype
    np.testing.assert_array_equal(actual, expected)
    if actual.dtype.kind == "f":
        bits = np.dtype(f"u{actual.dtype.itemsize}")
        finite_or_inf = ~np.isnan(expected)
        np.testing.assert_array_equal(actual.view(bits)[finite_or_inf], expected.view(bits)[finite_or_inf])


@pytest.mark.parametrize("operation", _OPERATIONS)
@pytest.mark.parametrize("source_dtype,lookup_dtype", [("f4", "f4"), ("f8", "f4"), ("f4", "f8"), ("f8", "f8")])
@pytest.mark.parametrize("spatial", [(), ("ncells",), ("lat", "lon"), ("depth", "lat", "lon")])
def test_gather_matches_groupby(operation, source_dtype, lookup_dtype, spatial):
    rng = np.random.default_rng(21)
    sizes = {"ncells": 7, "depth": 3, "lat": 5, "lon": 7}
    values = np.array([1, 3, np.nan, 2, 1, 37, 0, -2, 366, 367, 9, 366, 3])
    labels = [1, 3, 9, 366]
    shape = tuple(sizes[dim] for dim in spatial)
    source_data = rng.normal(size=(values.size, *shape)).astype(source_dtype)
    table_data = rng.normal(size=(len(labels), *shape)).astype(lookup_dtype)
    source_data.flat[0] = -0.0
    table_data.flat[0] = 0.0
    source_data.flat[-1] = np.nan
    table_data.flat[-1] = np.nan
    # Equal finite values include both tails, including when the operands differ in dtype.
    source_data[1] = table_data[1].astype("f4")
    table_data[1] = source_data[1]
    if operation == "divide":
        table_data[1] = 1e-11
    time = xr.date_range("2000-01-01", periods=values.size, freq="D").values.copy()
    time[2] = np.datetime64("NaT")
    time[4] = time[0]
    coords = {dim: np.arange(sizes[dim])[::-1] for dim in spatial}
    da = xr.DataArray(
        source_data,
        dims=("time", *spatial),
        coords={**coords, "time": time, "index": ("time", values), "offset": 2},
        name="sea_temperature",
        attrs={"units": "K"},
    )
    if spatial:
        da = da.transpose(spatial[0], "time", *spatial[1:])
    chunks = {"ncells": (3, 3, 1), "depth": (2, 1), "lat": (3, 2), "lon": (4, 3)}
    da = da.chunk({"time": (4, 3, 4, 2), **{dim: chunks[dim] for dim in spatial}})
    lookup = (
        xr.DataArray(table_data, dims=("index", *spatial), coords={**coords, "index": labels})
        .transpose(*reversed(spatial), "index")
        .chunk({"index": 2, **dict.fromkeys(spatial, 2)})
    )
    if operation == "divide":
        lookup = lookup.where(lookup > 1e-10)

    options = {"timedim": "time", "cycledim": "index", "operation": operation}
    expected = _legacy_apply(da, lookup, values, **options)
    actual = seasonal_apply.apply_seasonal_field(da, lookup, values, task_elements=56, **options)
    assert actual.chunks == da.chunks
    actual, expected = actual.compute(), expected.compute()
    _assert_bits(actual.values, expected.values)
    xr.testing.assert_identical(actual, expected)


@pytest.mark.parametrize(
    "calendar,freq,index_name",
    [("noleap", "D", "dayofyear"), ("360_day", "D", "dayofyear"), ("standard", "6h", "hourofyear"), ("360_day", "MS", "month")],
)
def test_calendar_labels(calendar, freq, index_name):
    time = xr.date_range("2000-01-01", periods=1501, freq=freq, calendar=calendar, use_cftime=True)
    da = xr.DataArray(np.arange(len(time), dtype="f4"), dims="time", coords={"time": time}).chunk({"time": 157})
    cycle = infer_cycle(da.time)
    assert cycle.index_name == index_name
    labels = cycle.labels[::3]
    lookup = xr.DataArray(labels.astype("f4"), dims=index_name, coords={index_name: labels}).chunk({index_name: 5})
    values = cycle.index_of(da.time).values
    options = {"timedim": "time", "cycledim": index_name, "operation": "subtract"}
    result = seasonal_apply.apply_seasonal_field(da, lookup, values, **options)
    assert result.chunks == da.chunks
    assert result.name is da.name
    _assert_bits(result.values, _legacy_apply(da, lookup, values, **options).values)


@pytest.mark.parametrize("calendar", ["noleap", "360_day", "all_leap"])
@pytest.mark.parametrize("source_dtype,lookup_dtype", [("f4", "f4"), ("f8", "f4"), ("i2", "f4"), ("i2", "i2")])
def test_divide_scrambled_calendar_labels(calendar, source_dtype, lookup_dtype):
    dates = xr.DataArray(
        xr.date_range("2000-12-26", periods=8, freq="D", calendar=calendar, use_cftime=True), dims="obs", name="date"
    )
    cycle = infer_cycle(dates)
    dates = dates.isel(obs=[5, 0, 0, 2, 7, 6, 3])
    values = cycle.index_of(dates).values
    labels = np.unique(np.r_[values[:4], values[5:]])[::-1]
    coords = {"lat": [1, 2], "lon": [3, 2, 1], "depth": [0, 5]}
    da = xr.DataArray(
        np.arange(84, dtype=source_dtype).reshape(2, 7, 3, 2),
        dims=("lat", "obs", "lon", "depth"),
        coords={**coords, "date": dates},
    ).chunk({"obs": (2, 1, 4), "lon": (2, 1)})
    lookup = xr.DataArray(
        np.arange(1, 12 * labels.size + 1, dtype=lookup_dtype).reshape(labels.size, 2, 3, 2),
        dims=("dayofyear", "lat", "lon", "depth"),
        coords={**coords, "dayofyear": labels},
    ).chunk({"dayofyear": 2, "depth": 1})

    actual = seasonal_apply.apply_seasonal_field(da, lookup, values, timedim="obs", cycledim="dayofyear", operation="divide")
    assert actual.chunks == da.chunks
    valid = np.isin(values, labels)
    selected = lookup.sel(dayofyear=xr.DataArray(values[valid], dims="obs")).transpose(*da.dims)
    quotient = np.true_divide(da.isel(obs=valid).values, selected.values)
    expected = np.full(da.shape, np.nan, dtype=quotient.dtype)
    expected[:, valid, :, :] = quotient
    _assert_bits(actual.values, expected)
    xr.testing.assert_identical(actual.compute(), da.copy(data=expected))


def test_eager_integer_divide_with_zero_divisors_and_invalid_labels():
    da = xr.DataArray(np.array([0, 1, -1, 8, 2, 3, 4], dtype="i2"), dims="t")
    lookup = xr.DataArray(np.array([2, 0, 1, 1], dtype="i2"), dims="slot", coords={"slot": [5, 2, 0, -1]})
    values = np.array([2, 2, 2, 5, 3, 0, -1])
    with np.errstate(divide="raise", invalid="raise"):
        actual = seasonal_apply.apply_seasonal_field(da, lookup, values, timedim="t", cycledim="slot", operation="divide")
    _assert_bits(actual.values, np.array([np.nan, np.inf, -np.inf, 4, np.nan, np.nan, np.nan], dtype="f8"))


@pytest.mark.parametrize("empty", [False, True])
def test_eager_lookup_positions(empty):
    labels = [] if empty else [366, 0, 1, 3, -1]
    da = xr.DataArray([0.0] * 8, dims="t")
    lookup = xr.DataArray(labels, dims="slot", coords={"slot": labels})
    values = np.array([1, 2, 3, 366, 367, np.datetime64("NaT"), -1, 0], dtype=object)
    result = seasonal_apply.apply_seasonal_field(da, lookup, values, timedim="t", cycledim="slot", operation="subtract")
    expected = [np.nan] * 8 if empty else [-1, np.nan, -3, -366, np.nan, np.nan, np.nan, np.nan]
    assert result.chunks is None
    np.testing.assert_array_equal(result.values, expected)


def test_executed_blocks_fit_task_budget(monkeypatch):
    budget = 39
    source = xr.DataArray(dsa.ones((17, 3, 5, 11), chunks=(9, 2, 4, 6)), dims=("time", "depth", "lat", "lon"))
    lookup = xr.DataArray(
        dsa.ones((7, 3, 5, 11), chunks=(2, 3, 5, 11)),
        dims=("slot", "depth", "lat", "lon"),
        coords={"slot": np.arange(1, 8)},
    )
    apply_block = seasonal_apply._apply_block
    evaluated = []

    def checked(source, lookup, positions, **kwargs):
        evaluated.append(lookup.shape)
        assert lookup.shape[0] == 7
        assert source.size <= budget
        assert lookup.size <= budget
        return apply_block(source, lookup, positions, **kwargs)

    monkeypatch.setattr(seasonal_apply, "_apply_block", checked)
    values = 1 + np.arange(source.sizes["time"]) % 7
    result = seasonal_apply.apply_seasonal_field(
        source, lookup, values, timedim="time", cycledim="slot", operation="subtract", task_elements=budget
    )
    assert result.chunks == source.chunks
    assert not evaluated
    np.testing.assert_array_equal(result.compute(scheduler="synchronous").values, np.zeros(source.shape))
    assert evaluated


def test_fitting_anisotropic_tiles_stay_whole(monkeypatch):
    source = xr.DataArray(dsa.ones((3, 8, 21), chunks=(3, 8, 1)), dims=("time", "lat", "lon"))
    lookup = xr.DataArray(dsa.ones((4, 8, 21), chunks=(2, 8, 1)), dims=("slot", "lat", "lon"), coords={"slot": [1, 2, 3, 4]})
    apply_block = seasonal_apply._apply_block

    def checked(source, lookup, positions, **kwargs):
        assert lookup.shape == (4, 8, 1)
        return apply_block(source, lookup, positions, **kwargs)

    monkeypatch.setattr(seasonal_apply, "_apply_block", checked)
    result = seasonal_apply.apply_seasonal_field(
        source, lookup, np.array([3, 1, 4]), timedim="time", cycledim="slot", operation="subtract", task_elements=32
    )
    np.testing.assert_array_equal(result.values, np.zeros(source.shape))


def test_graph_size_across_200_million_lookup_elements():
    counts = []
    cells = 100_000
    for steps in (511, 1023):
        source = xr.DataArray(dsa.empty((steps, cells + 5), chunks=(17, cells)), dims=("time", "ncells"))
        by_cycle = []
        for n_cycle in (1999, 2000, 2001):
            lookup = xr.DataArray(
                dsa.empty((n_cycle, cells + 5), chunks=(97, 25_000)),
                dims=("slot", "ncells"),
                coords={"slot": np.arange(1, n_cycle + 1)},
            )
            values = 1 + np.arange(steps) * 59 % n_cycle
            result = seasonal_apply.apply_seasonal_field(source, lookup, values, timedim="time", cycledim="slot", operation="upper")
            assert result.chunks == source.chunks
            by_cycle.append(len(result.data.__dask_graph__()))
        counts.append(by_cycle)
    assert max(counts[1]) < 2_000
    assert max(counts[1]) < 1.5 * min(counts[1])
    assert all(long <= 2 * short for short, long in zip(*counts))


@pytest.mark.parametrize("tail", ["upper", "lower"])
@pytest.mark.parametrize("compute_mode", ["persist", "lazy", "streaming"])
def test_pipeline_matches_legacy_gather(compute_mode, tail, monkeypatch, tmp_path):
    time = xr.date_range("1999-01-01", periods=27, freq="MS", calendar="360_day", use_cftime=True)
    base = np.sin(np.arange(time.size) * np.pi / 6)[:, None, None] + np.arange(time.size)[:, None, None] / 10
    data = (base + np.arange(6).reshape(2, 3)).astype("f4")
    da = xr.DataArray(data, dims=("time", "lat", "lon"), coords={"time": time, "lat": [20, 10], "lon": [0, 2, 4]}).chunk(
        {"time": 5, "lat": 1, "lon": 2}
    )
    options = {
        "method_anomaly": "fixed_baseline",
        "reference_period": (1999, 1999),
        "method_extreme": "seasonal_percentile",
        "method_percentile": "exact",
        "smooth_days": 60,
        "window_days": 30,
        "dask_chunks": {"time": 7},
        "tail": tail,
        "threshold_percentile": 90 if tail == "upper" else 10,
    }
    with monkeypatch.context() as patch:
        patch.setattr(fixed_baseline, "apply_seasonal_field", _legacy_apply)
        patch.setattr(seasonal_percentile, "apply_seasonal_field", _legacy_apply)
        expected = marEx.preprocess_data(da, compute_mode="lazy", **options).compute()
    result = marEx.preprocess_data(da, compute_mode=compute_mode, scratch_dir=str(tmp_path), **options)
    try:
        for name in ("dat_anomaly", "extreme_events"):
            assert result[name].chunks == ((7, 7, 7, 6), (2,), (3,))
        assert result.thresholds.chunks == ((7, 5), (2,), (3,))
        computed = result.compute()
        for name in expected.data_vars:
            _assert_bits(computed[name].values, expected[name].values)
            xr.testing.assert_identical(computed[name], expected[name])
        if compute_mode == "streaming":
            staging = Path(result.encoding["marex_staging_dir"])
            assert (staging / "thresholds.zarr").exists()
            assert "marex_staging_dir" not in result.attrs
            out = tmp_path / "out.zarr"
            result.to_zarr(out)
            marEx.clear_staging(result)
            assert not staging.exists()
            with xr.open_zarr(out) as reopened:
                xr.testing.assert_identical(reopened, computed)
    finally:
        if compute_mode == "streaming" and Path(result.encoding["marex_staging_dir"]).exists():
            marEx.clear_staging(result)


@pytest.mark.parametrize("single", [True, False])
def test_fixed_baseline_scalar_and_float64(single, monkeypatch):
    time = np.array(["1999-12-29", "1999-12-30", "1999-12-31", "2000-12-29", "2000-12-30", "2000-12-31"], dtype="datetime64[D]")
    data = 2.0**24 + np.array([0.0, 1.25, 4.0, 1.75, 5.5, 7.0])
    if single:
        time, data = time[:1], data[:1]
    da = xr.DataArray(data, dims="time", coords={"time": time}).chunk({"time": 4})
    options = {
        "dimensions": {"time": "time"},
        "reference_period": (1999, 1999),
        "cycle": SeasonalCycle("dayofyear", 366, 1),
        "smooth_days": 3,
    }
    with monkeypatch.context() as patch:
        patch.setattr(fixed_baseline, "apply_seasonal_field", _legacy_apply)
        expected = fixed_baseline._compute_anomaly_fixed_baseline(da, **options).compute()
    result = fixed_baseline._compute_anomaly_fixed_baseline(da, **options)
    assert result.dat_anomaly.chunks == da.chunks
    assert result.dat_anomaly.dtype == np.float32
    assert result.mask.ndim == 0
    _assert_bits(result.dat_anomaly.values, expected.dat_anomaly.values)


def test_source_missing_value_attr_does_not_corrupt_streaming_or_flags(tmp_path):
    da = xr.DataArray(
        np.tile(np.arange(1, 13, dtype="f4"), 2)[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": xr.date_range("2001", periods=24, freq="MS"), "lon": [0], "lat": [0]},
        attrs={"missing_value": np.float32(0), "units": "K"},
        name="tas",
    ).chunk({"time": (3, 7, 4, 10)})
    opts = {
        "method_anomaly": "fixed_baseline",
        "method_extreme": "seasonal_percentile",
        "method_percentile": "exact",
        "smooth_days": 1,
        "window_days": 30,
        "dask_chunks": {"time": 5},
    }
    for mode in ("lazy", "streaming"):
        ds = marEx.preprocess_data(da, compute_mode=mode, scratch_dir=str(tmp_path), **opts)
        try:
            assert int(np.isfinite(ds.dat_anomaly.values).sum()) == 24
            assert int((ds.dat_anomaly.values == 0).sum()) == 24
            assert int(np.isfinite(ds.thresholds.values).sum()) == 12
            store = tmp_path / f"{mode}.zarr"
            ds.to_zarr(store)
            if mode == "streaming":
                marEx.clear_staging(ds)
            with xr.open_zarr(store) as reopened:
                assert int(np.isfinite(reopened.dat_anomaly.values).sum()) == 24
        finally:
            ticket = ds.encoding.get("marex_staging_dir")
            if ticket and Path(ticket).exists():
                marEx.clear_staging(ds)

    anom = xr.DataArray(
        np.tile(np.array([-1.0, 1.0], dtype="f4"), 12)[:, None, None],
        dims=("time", "lat", "lon"),
        coords={"time": xr.date_range("2001", periods=24, freq="MS"), "lon": [0], "lat": [0]},
        attrs={"missing_value": np.float32(0)},
    ).chunk({"time": 6})
    flags, _ = marEx.extremes.identify_extremes(
        anom, method_extreme="seasonal_percentile", method_percentile="exact", window_days=30
    )
    assert "missing_value" not in flags.attrs
    flag_store = tmp_path / "flags.zarr"
    flags.rename("flags").to_zarr(flag_store)
    with xr.open_zarr(flag_store) as reopened_flags:
        np.testing.assert_array_equal(reopened_flags.flags.values, flags.values)


@pytest.mark.parametrize("chunked", [False, True])
def test_int64_adjacent_subtraction_with_missing_label_and_null_lookup_labels(chunked):
    import pandas as pd

    base = 2**60
    a = xr.DataArray(np.array([base + 1, base + 2], dtype="i8"), dims="t")
    b = xr.DataArray(np.array([base], dtype="i8"), dims="s", coords={"s": [1]})
    if chunked:
        a = a.chunk({"t": 1})
    full = seasonal_apply.apply_seasonal_field(a, b, np.array([1, 1]), timedim="t", cycledim="s", operation="subtract")
    assert full.dtype == np.int64
    np.testing.assert_array_equal(full.values, np.array([1, 2], dtype="i8"))

    gapped = seasonal_apply.apply_seasonal_field(a, b, np.array([1, 0]), timedim="t", cycledim="s", operation="subtract")
    assert gapped.dtype == np.float64
    np.testing.assert_array_equal(gapped.values, np.array([1.0, np.nan]))

    src = xr.DataArray(np.ones(2, dtype="f4"), dims="t")
    for sentinel in (None, pd.NA, np.nan):
        lut = xr.DataArray(np.ones(2, dtype="f4"), dims="s", coords={"s": np.array([1, sentinel], dtype=object)})
        res = seasonal_apply.apply_seasonal_field(
            src, lut, np.array([1, sentinel], dtype=object), timedim="t", cycledim="s", operation="subtract"
        )
        np.testing.assert_array_equal(res.values, np.array([0.0, np.nan], dtype="f4"))


def test_preprocess_data_with_non_index_time_coordinate():
    dates = xr.date_range("2001-01-01", periods=24, freq="MS")
    da = xr.DataArray(
        np.tile(np.arange(1, 13, dtype="f4"), 2)[:, None, None],
        dims=("record", "lat", "lon"),
        coords={"valid_at": ("record", dates.values), "lat": [0.0], "lon": [0.0]},
    ).chunk({"record": 6})
    ds = marEx.preprocess_data(
        da,
        method_anomaly="fixed_baseline",
        method_extreme="seasonal_percentile",
        method_percentile="exact",
        smooth_days=1,
        window_days=30,
        dimensions={"time": "record", "y": "lat", "x": "lon"},
        coordinates={"time": "valid_at", "y": "lat", "x": "lon"},
        compute_mode="lazy",
    ).compute()
    assert int(np.isfinite(ds.dat_anomaly.values).sum()) == 24

    # Indexed integer record dimension + separate datetime valid_at coordinate on harmonic & detrend_fixed_baseline
    da_idx = da.assign_coords(record=np.arange(24) + 900)
    for method in ("detrend_harmonic", "detrend_fixed_baseline"):
        ds_h = marEx.preprocess_data(
            da_idx,
            method_anomaly=method,
            standardise=(method == "detrend_harmonic"),
            method_extreme="seasonal_percentile",
            method_percentile="exact",
            smooth_days=1,
            window_days=30,
            dimensions={"time": "record", "y": "lat", "x": "lon"},
            coordinates={"time": "valid_at", "y": "lat", "x": "lon"},
            compute_mode="lazy",
        ).compute()
        da_canon = da.rename({"valid_at": "time"}).swap_dims({"record": "time"})
        ds_ref = marEx.preprocess_data(
            da_canon,
            method_anomaly=method,
            standardise=(method == "detrend_harmonic"),
            method_extreme="seasonal_percentile",
            method_percentile="exact",
            smooth_days=1,
            window_days=30,
            compute_mode="lazy",
        ).compute()
        for var in ("dat_anomaly", "extreme_events"):
            np.testing.assert_array_equal(ds_h[var].values, ds_ref[var].values)
