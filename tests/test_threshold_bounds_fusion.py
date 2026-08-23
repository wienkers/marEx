"""Guard the threshold bounds check against re-walking the anomaly graph.

``if too_high.any():`` invokes ``DataArray.__bool__``, which computes. Before the fusion
this path made up to four separate scheduler round-trips per threshold -- an implicit
compute per predicate, plus a ``.max()``/``.min()`` inside each warning body -- and every
one of them re-executes the whole upstream anomaly graph whenever the anomaly is not
pinned (which is exactly the case in the lazy and streaming compute modes).

Values are unaffected by the fusion. What this module pins is the *graph traversal count*,
which is precisely the class of regression that bit-identity tests are blind to.
"""

import numpy as np
import pandas as pd
import pytest
import xarray as xr
from dask.callbacks import Callback

from marEx.extremes.histogram import _compute_histogram_quantile_1d, _compute_histogram_quantile_2d


class CountComputes(Callback):
    """Count top-level dask scheduler round-trips (compute and persist both fire ``_start``)."""

    def __init__(self):
        self.n = 0

    def _start(self, dsk):
        self.n += 1


def _anomaly_fixture(n_time=400, n_y=6, n_x=7, seed=0, constant_corner=False):
    """A small dask-backed anomaly array with a full-length time axis."""
    rng = np.random.default_rng(seed)
    data = rng.normal(0.0, 1.0, size=(n_time, n_y, n_x)).astype(np.float32)
    if constant_corner:
        # A constant-zero cell drives its quantile below bin_edges[3], exercising the clamp.
        data[:, 0, 0] = 0.0
    da = xr.DataArray(
        data,
        dims=("time", "lat", "lon"),
        coords={
            "time": pd.date_range("2000-01-01", periods=n_time, freq="D"),
            "lat": np.arange(n_y, dtype=np.float32),
            "lon": np.arange(n_x, dtype=np.float32),
        },
        name="da",
    )
    return da.chunk({"time": -1, "lat": 3, "lon": 7})


def _default_bin_edges(precision=0.01, max_anomaly=5.0):
    return np.concatenate([[-np.inf], np.arange(-precision, max_anomaly + precision, precision)])


class TestBoundsCheckRoundTrips:
    """The bounds check must cost one fused round-trip, not one per predicate."""

    def test_1d_bounds_check_makes_at_most_two_round_trips(self):
        """Measured 5 originally, 4 after the bounds fusion, 2 now.

        The two that remain are the threshold persist and one fused bounds check. The two
        that went are the CDF persist and the ``idx_upper`` compute, both removed when the
        1D quantile was fused into a single ``apply_ufunc`` -- together they were the
        ``space x n_bins`` CDF ceiling.
        """
        da = _anomaly_fixture()
        with CountComputes() as cb:
            _compute_histogram_quantile_1d(da, q=0.95, dim="time")
        assert cb.n <= 2, f"expected <= 2 scheduler round-trips, got {cb.n}"

    def test_1d_makes_a_single_round_trip_in_lazy_mode(self):
        """In lazy mode nothing is pinned, so only the fused bounds check remains.

        This is the assertion that would catch a re-introduced CDF persist: it is
        space-scaled (~30 GB on the ICON mesh) and no amount of time-chunking shrinks it,
        so it must not come back under any compute_mode.
        """
        from marEx.core.compute_mode import Materialiser

        da = _anomaly_fixture()
        with CountComputes() as cb:
            _compute_histogram_quantile_1d(da, q=0.95, dim="time", materialiser=Materialiser("lazy"))
        assert cb.n <= 1, f"expected <= 1 scheduler round-trip in lazy mode, got {cb.n}"

    def test_2d_bounds_check_makes_at_most_four_round_trips(self):
        """Measured 5 before the fusion, 4 after.

        The 2D path computes its ``nan_mask`` separately (it cannot derive the mask from
        histogram totals the way the 1D path does, because the spatial rolling mixes
        neighbouring cells' counts) and persists the threshold.
        """
        da = _anomaly_fixture()
        da = da.assign_coords(dayofyear=da.time.dt.dayofyear)
        with CountComputes() as cb:
            _compute_histogram_quantile_2d(
                da,
                q=0.95,
                dimensions={"time": "time", "x": "lon", "y": "lat"},
                window_days=3,
                window_spatial=None,
            )
        assert cb.n <= 4, f"expected <= 4 scheduler round-trips, got {cb.n}"


class TestBoundsCheckBehaviourUnchanged:
    """Fusing the computes must not change a single observable behaviour."""

    def test_still_warns_when_threshold_exceeds_the_top_bin(self):
        da = _anomaly_fixture()
        # max_anomaly small enough that the 95th percentile lands above the top bin edge.
        with pytest.warns(UserWarning, match="exceed expected range"):
            _compute_histogram_quantile_1d(da, q=0.95, dim="time", precision=0.01, max_anomaly=0.5)

    def test_still_warns_and_clamps_when_threshold_is_below_the_lower_bound(self):
        da = _anomaly_fixture(constant_corner=True)
        with pytest.warns(UserWarning, match="below expected range"):
            result = _compute_histogram_quantile_1d(da, q=0.95, dim="time")
        lower_bound = float(_default_bin_edges()[3])
        assert float(result.isel(lat=0, lon=0).values) == pytest.approx(lower_bound)

    def test_no_warning_for_a_well_scaled_field(self):
        """A field comfortably inside the bins must not warn at all."""
        da = _anomaly_fixture()
        with warnings_as_errors():
            _compute_histogram_quantile_1d(da, q=0.95, dim="time")

    def test_values_match_a_direct_numpy_percentile(self):
        """The fusion must not touch the returned quantiles."""
        da = _anomaly_fixture()
        result = _compute_histogram_quantile_1d(da, q=0.95, dim="time").compute()
        reference = np.percentile(da.compute().values, 95, axis=0)
        # The histogram method is approximate at the bin precision (0.01).
        np.testing.assert_allclose(result.values, reference, atol=0.05)


class warnings_as_errors:
    """Context manager turning UserWarning into an error, for the no-warning case."""

    def __enter__(self):
        import warnings

        self._ctx = warnings.catch_warnings()
        self._ctx.__enter__()
        warnings.simplefilter("error", UserWarning)
        return self

    def __exit__(self, *exc):
        return self._ctx.__exit__(*exc)
