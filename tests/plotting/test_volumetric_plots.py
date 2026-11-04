"""Plot helpers for volumetric events: each figure is checked against the numbers in the events dataset.

A figure that renders is not evidence it shows the right thing, so the map and section tests read the
drawn arrays back off the axes and compare them with the ID field. Draw tests also count which time
steps the lazy ID field was asked for, so a helper that loads the whole field to draw one panel fails.
The animation tests stub only the encoder; the frames are drawn for real, and every frame's pixel size
is read back.
"""

from __future__ import annotations

import shutil
import warnings
from pathlib import Path

import dask
import dask.array as dsa
import matplotlib

matplotlib.use("Agg")

import cartopy.crs as ccrs  # noqa: E402
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pytest  # noqa: E402
import xarray as xr  # noqa: E402
from PIL import Image  # noqa: E402

import marEx  # noqa: E402
from marEx.exceptions import ConfigurationError  # noqa: E402
from marEx.plotX import (  # noqa: E402
    PlotConfig,
    animate_event_levels,
    plot_event_depth_time,
    plot_event_levels,
    plot_event_summary,
)
from marEx.plotX import animation as animation_mod  # noqa: E402
from marEx.plotX import volumetric as vol_mod  # noqa: E402

from ..track.volumetric.synthetic import make_inputs, scenarios  # noqa: E402

ZDIM = "depth"
DIMS = {"time": "time", "z": "depth", "y": "lat", "x": "lon"}


@pytest.fixture(autouse=True)
def _close_figures():
    yield
    plt.close("all")


def _run(name):
    binary, meta = scenarios()[name]
    extremes, wet, bounds, areas = make_inputs(binary, **meta["make_inputs"])
    tracker = marEx.tracker(
        extremes,
        wet,
        layer_bounds=bounds,
        cell_areas=areas,
        R_fill=0,
        T_fill=0,
        dimensions=DIMS,
        nn_partitioning=True,
        regional_mode=True,
        coordinate_units="degrees",
        volume_filter_absolute=1.0,
        quiet=True,
    )
    events, merges = tracker.run(return_merges=True)
    cell_volume = ((bounds[:, 1] - bounds[:, 0]) * areas).transpose("depth", "lat", "lon").compute()
    # Numpy-backed so no test depends on dask state.
    return events.compute(), merges.compute(), cell_volume


@pytest.fixture(scope="module")
def merge_run():
    return _run("merge_then_split")


@pytest.fixture(scope="module")
def birth_run():
    return _run("birth_death")


@pytest.fixture(scope="module")
def grow_run():
    return _run("persist_grow_decay")


def _mesh_values(ax):
    """The array drawn by the first mesh on ``ax`` (cartopy maps and plain axes alike)."""
    return np.ma.filled(np.ma.masked_invalid(ax.collections[0].get_array()), np.nan)


def _map_axes(fig):
    return [a for a in fig.axes if hasattr(a, "projection")]


class _Counting:
    """Array-like that records which time steps dask reads from it."""

    def __init__(self, values):
        self.values = values
        self.shape, self.dtype, self.ndim = values.shape, values.dtype, values.ndim
        self.times = set()

    def __getitem__(self, key):
        t = key[0]
        self.times.update(range(*t.indices(self.shape[0])) if isinstance(t, slice) else [t])
        return self.values[key]


def _lazy_events(events):
    """Same dataset with the ID field behind a counting reader, one time step per chunk."""
    src = _Counting(events["ID_field"].values)
    lazy = xr.DataArray(
        dsa.from_array(src, chunks=(1,) + src.shape[1:], asarray=False),
        dims=events["ID_field"].dims,
        coords=events["ID_field"].coords,
    )
    return events.assign(ID_field=lazy), src


# -- 1. map panels ------------------------------------------------------------------------------


def _differing_levels(events):
    """A time step and levels whose occupied cells differ from one another, so a panel drawn from the wrong level shows."""
    ids = events["ID_field"].transpose("time", ZDIM, "lat", "lon").values
    for t in range(ids.shape[0]):
        occupied = [k for k in range(ids.shape[1]) if (ids[t, k] > 0).any()]
        picks = [k for k in occupied if all(not np.array_equal(ids[t, k] > 0, ids[t, j] > 0) for j in occupied if j < k)]
        if len(picks) >= 2:
            return t, picks, ids[t]
    raise AssertionError("fixture has no time step with levels of different content")


def test_levels_title_uses_whole_metres_from_ten_up_and_one_decimal_below(grow_run):
    events, _, _ = grow_run
    t, levels, _ = _differing_levels(events)
    long_floats = events.assign_coords({ZDIM: 1234.5678 + np.arange(events.sizes[ZDIM], dtype=float)})
    fig = plot_event_levels(long_floats, t, levels[:1], zdim=ZDIM)
    assert _map_axes(fig)[0].get_title() == f"{ZDIM} = {1234.5678 + levels[0]:.0f} m"
    # Near the surface the level is a fraction of a metre, so one decimal is kept.
    shallow = events.assign_coords({ZDIM: 0.494 + 0.5 * np.arange(events.sizes[ZDIM], dtype=float)})
    fig = plot_event_levels(shallow, t, levels[:1], zdim=ZDIM)
    assert _map_axes(fig)[0].get_title() == f"{ZDIM} = {0.494 + 0.5 * levels[0]:.1f} m"


def test_levels_extent_zooms_the_maps_and_the_event(merge_run):
    events, _, _ = merge_run
    t, levels, ids = _differing_levels(events)
    ev = int(ids[levels[0]][ids[levels[0]] > 0][0])
    lons, lats = events["lon"].values, events["lat"].values
    flat = ccrs.PlateCarree()  # a rectangular projection, so the requested window is the drawn window
    whole = _map_axes(plot_event_levels(events, t, levels[:1], zdim=ZDIM, projection=flat))[0].get_extent(ccrs.PlateCarree())
    chosen = (lons[1], lons[-2], lats[1], lats[-2])
    drawn = _map_axes(plot_event_levels(events, t, levels[:1], zdim=ZDIM, extent=chosen, projection=flat))[0].get_extent(
        ccrs.PlateCarree()
    )
    np.testing.assert_allclose(drawn, chosen, atol=1e-6)
    assert not np.allclose(drawn, whole), "the extent must change the window"
    # Zooming to the event frames its footprint on the drawn levels and nothing else is asked of the caller.
    member = (ids[levels] == ev).any(axis=0)
    rows, cols = np.flatnonzero(member.any(axis=1)), np.flatnonzero(member.any(axis=0))
    zoomed = _map_axes(plot_event_levels(events, t, levels, zdim=ZDIM, event_id=ev, zoom_to_event=True, projection=flat))[
        0
    ].get_extent(ccrs.PlateCarree())
    footprint = (lons[cols[0]], lons[cols[-1]], lats[rows[0]], lats[rows[-1]])
    assert zoomed[0] <= footprint[0] and zoomed[1] >= footprint[1] and zoomed[2] <= footprint[2] and zoomed[3] >= footprint[3]
    assert (zoomed[1] - zoomed[0]) < (whole[1] - whole[0]) or (zoomed[3] - zoomed[2]) < (whole[3] - whole[2])
    # An explicit extent wins over the event, and bad arguments raise.
    both = _map_axes(
        plot_event_levels(events, t, levels, zdim=ZDIM, event_id=ev, zoom_to_event=True, extent=chosen, projection=flat)
    )[0].get_extent(ccrs.PlateCarree())
    np.testing.assert_allclose(both, chosen, atol=1e-6)
    with pytest.raises(ConfigurationError, match="event_id"):
        plot_event_levels(events, t, levels, zdim=ZDIM, zoom_to_event=True)
    for bad in ((1, 2, 3), (5, 1, 0, 1), "xyzw"):
        with pytest.raises(ConfigurationError):
            plot_event_levels(events, t, levels, zdim=ZDIM, extent=bad)


def _hand_events():
    """Event 1 on a 1-degree grid: level 0 holds cols 2-4 and rows 5-6, level 1 holds cols 20-22 and rows 7-8."""
    ids = np.zeros((1, 3, 20, 40), dtype=np.int32)
    ids[0, 0, 5:7, 2:5] = 1
    ids[0, 1, 7:9, 20:23] = 1
    coords = {
        "time": np.array(["2000-01-01"], dtype="datetime64[ns]"),
        ZDIM: [5.0, 50.0, 500.0],
        "lat": np.arange(20.0),
        "lon": np.arange(40.0),
    }
    return xr.Dataset({"ID_field": ((("time", ZDIM, "lat", "lon")), ids)}, coords={**coords, "ID": [1]})


def test_levels_zoom_to_event_frames_the_union_of_the_drawn_levels_with_a_margin():
    events = _hand_events()
    flat = ccrs.PlateCarree()
    # Footprint over both levels: lon 2..22 (span 20), lat 5..8 (span 3). The margin is 15% of the larger span, 3 degrees.
    both = plot_event_levels(events, 0, [0, 1], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=flat)
    np.testing.assert_allclose(_map_axes(both)[0].get_extent(flat), (-1.0, 25.0, 2.0, 11.0), atol=1e-6)
    # Only level 1 drawn: lon 20..22 (span 2), lat 7..8, and the margin is at least 2 degrees.
    deep = plot_event_levels(events, 0, [1], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=flat)
    np.testing.assert_allclose(_map_axes(deep)[0].get_extent(flat), (18.0, 24.0, 5.0, 10.0), atol=1e-6)


def _module_log(caplog):
    """The package logger does not propagate, so hand caplog's handler to this module's logger."""
    vol_mod.logger.addHandler(caplog.handler)
    caplog.set_level("WARNING", logger=vol_mod.logger.name)


def _warnings(caplog):
    return [r.getMessage() for r in caplog.records if r.levelname == "WARNING"]


def _global_events(columns, rows=(9, 10)):
    """Event 1 on a global 1-degree -180..179 grid, in the given columns and rows of level 0."""
    lon, lat = np.arange(-180.0, 180.0), np.arange(-90.0, 91.0)
    ids = np.zeros((1, 2, lat.size, lon.size), dtype=np.int32)
    ids[0, 0, rows[0] : rows[1] + 1, list(columns)] = 1
    coords = {"time": np.array(["2000-01-01"], dtype="datetime64[ns]"), ZDIM: [5.0, 50.0], "lat": lat, "lon": lon}
    return xr.Dataset({"ID_field": ((("time", ZDIM, "lat", "lon")), ids)}, coords={**coords, "ID": [1]})


def test_levels_zoom_on_an_event_absent_there_shows_the_whole_domain_and_warns(caplog):
    _module_log(caplog)
    events = _hand_events()
    flat = ccrs.PlateCarree()
    full = _map_axes(plot_event_levels(events, 0, [2], zdim=ZDIM, event_id=1, projection=flat))[0].get_extent(flat)
    assert _warnings(caplog) == []
    zoomed = _map_axes(plot_event_levels(events, 0, [2], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=flat))[0]
    np.testing.assert_allclose(zoomed.get_extent(flat), full, atol=1e-6)
    assert len(_warnings(caplog)) == 1 and "Event 1" in _warnings(caplog)[0] and "whole domain" in _warnings(caplog)[0]


def test_levels_zoom_on_an_event_present_there_stays_quiet(caplog):
    _module_log(caplog)
    plot_event_levels(_hand_events(), 0, [0, 1], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=ccrs.PlateCarree())
    assert _warnings(caplog) == []


def test_levels_zoom_on_an_event_across_the_seam_keeps_it_in_view_and_warns(caplog):
    _module_log(caplog)
    flat = ccrs.PlateCarree()
    events = _global_events([*range(0, 6), *range(354, 360)])
    extent = _map_axes(plot_event_levels(events, 0, [0], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=flat))[0]
    west, east, *_ = extent.get_extent(flat)
    assert west <= -180.0 + 1e-6 and east >= 179.0 - 1e-6
    assert any("seam" in message and "central_longitude" in message for message in _warnings(caplog))


def test_levels_zoom_across_the_seam_of_a_coarse_global_grid_warns_once(caplog):
    _module_log(caplog)
    flat = ccrs.PlateCarree()
    # At 2 degrees the event spans 358 degrees between the grid's edge columns, short of the 359 that fills the width.
    events = _global_events([*range(0, 6), *range(354, 360)]).isel(lon=slice(None, None, 2))
    west, east, *_ = _map_axes(plot_event_levels(events, 0, [0], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=flat))[
        0
    ].get_extent(flat)
    assert west <= -180.0 + 1e-6 and east >= 178.0 - 1e-6
    messages = _warnings(caplog)
    assert len(messages) == 1 and "seam" in messages[0]


def test_levels_zoom_on_a_wide_global_event_stays_on_the_grid(caplog):
    _module_log(caplog)
    flat = ccrs.PlateCarree()
    events = _global_events(range(10, 250), rows=(1, 179))
    ax = _map_axes(plot_event_levels(events, 0, [0], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=flat))[0]
    west, east, south, north = ax.get_extent(flat)
    # Lon -170..69 with a margin of 15% of 239 degrees, 35.85, would run past -180 and wrap.
    np.testing.assert_allclose((west, east), (-180.0, 104.85), atol=1e-6)
    assert south >= -90.0 - 1e-6 and north <= 90.0 + 1e-6
    assert _warnings(caplog) == []
    # A margin past the poles makes Robinson's axis limits NaN, and set_extent raises.
    plot_event_levels(events, 0, [0], zdim=ZDIM, event_id=1, zoom_to_event=True)


@pytest.mark.parametrize(
    "extent, projection, warned",
    [
        ((170.0, 190.0), ccrs.Robinson(), True),
        ((170.0, 190.0), ccrs.PlateCarree(), True),
        ((-190.0, -170.0), ccrs.PlateCarree(), True),
        ((170.0, 190.0), ccrs.PlateCarree(central_longitude=180), False),
        ((-10.0, 10.0), ccrs.PlateCarree(central_longitude=180), True),
        ((0.0, 30.0), ccrs.Robinson(), False),
        ((-180.0, 180.0), ccrs.Robinson(), False),
        ((200.0, 250.0), ccrs.Robinson(), False),
    ],
)
def test_levels_extent_across_the_projection_seam_warns(caplog, extent, projection, warned):
    _module_log(caplog)
    plot_event_levels(_hand_events(), 0, [0], zdim=ZDIM, extent=(*extent, 0.0, 20.0), projection=projection)
    assert any("seam" in message for message in _warnings(caplog)) is warned


@pytest.mark.parametrize(
    "extent, projection, warned",
    [
        # On the seam, not only across it.
        ((180.0, 200.0, 0.0, 20.0), ccrs.PlateCarree(), True),
        ((180.0, 200.0, 0.0, 20.0), ccrs.Robinson(), True),
        ((180.0, 360.0, 0.0, 20.0), ccrs.PlateCarree(central_longitude=180), True),
        ((-180.0, -170.0, 0.0, 20.0), ccrs.PlateCarree(), False),
        ((-180.0, -170.0, 0.0, 20.0), ccrs.Robinson(), False),
        # A whole turn off the map's own range collapses instead of widening.
        ((0.0, 360.0, 0.0, 20.0), ccrs.PlateCarree(), True),
        ((-20.0, 340.0, 0.0, 20.0), ccrs.PlateCarree(), True),
        ((0.0, 360.0, 0.0, 20.0), ccrs.PlateCarree(central_longitude=180), False),
        # Wide, across the seam, and wide, inside it.
        ((-100.0, 200.0, 0.0, 20.0), ccrs.PlateCarree(), True),
        ((-100.0, 200.0, 0.0, 20.0), ccrs.PlateCarree(central_longitude=50), False),
        ((-170.0, 160.0, 0.0, 20.0), ccrs.PlateCarree(), False),
        # A pseudocylindrical map is narrower at high latitude, so its full width there is well short of x_limits.
        ((170.0, 190.0, 50.0, 70.0), ccrs.Robinson(), True),
        ((180.0, 200.0, 40.0, 60.0), ccrs.Robinson(), True),
        ((-100.0, 200.0, 50.0, 70.0), ccrs.Robinson(), True),
        ((170.0, 190.0, 50.0, 70.0), ccrs.Mollweide(), True),
        ((10.0, 50.0, 50.0, 70.0), ccrs.Robinson(), False),
        # Across the seam on a band over the equator: about the width of a framed window, in the wrong place.
        ((-100.0, 200.0, -60.0, 60.0), ccrs.Robinson(), True),
        ((-122.0, 178.0, -60.0, 60.0), ccrs.Robinson(central_longitude=180), True),
        ((-100.0, 200.0, -43.0, 80.0), ccrs.Mollweide(), True),
        # Ending on the seam can frame, and then it stays quiet.
        ((180.0, 200.0, -10.0, 10.0), ccrs.Robinson(), False),
        # Asking for the whole width and getting it.
        ((0.0, 359.0, 0.0, 20.0), ccrs.PlateCarree(), False),
        ((-170.0, 160.0, 50.0, 70.0), ccrs.Robinson(), False),
        # A narrow window is small, not collapsed.
        ((10.0, 10.2, 0.0, 1.0), ccrs.PlateCarree(), False),
        # Polar maps have no seam, and a sector past 180 degrees spans their whole width.
        ((170.0, 190.0, 40.0, 60.0), ccrs.NorthPolarStereo(), False),
        ((0.0, 300.0, 40.0, 60.0), ccrs.NorthPolarStereo(), False),
        # An off-axis polar sector is narrower than the centred one, without having collapsed.
        ((85.0, 95.0, 60.0, 62.0), ccrs.NorthPolarStereo(), False),
        ((30.0, 150.0, 59.0, 60.0), ccrs.NorthPolarStereo(), False),
        # A polar sector can draw a percent wider than the centred one.
        ((-326.0, -86.0, 60.0, 85.0), ccrs.NorthPolarStereo(), False),
        ((-120.0, -70.0, 25.0, 50.0), ccrs.LambertConformal(central_longitude=-96), False),
        # Ends on the seam and spreads across it, drawing the full width, only 1.2 times the framed one.
        ((-480.0, -180.0, 0.0, 20.0), ccrs.PlateCarree(), True),
        # Past a whole turn: these crop to 320, 340 and 358 degrees, while the next ones draw the full window.
        ((-200.0, 200.0, 0.0, 20.0), ccrs.PlateCarree(), True),
        ((-190.0, 190.0, 0.0, 20.0), ccrs.PlateCarree(), True),
        ((-181.0, 181.0, 0.0, 20.0), ccrs.PlateCarree(), True),
        ((-0.05, 359.95000000000005, 0.0, 20.0), ccrs.PlateCarree(), False),
        ((-180.0, 180.5, 0.0, 20.0), ccrs.PlateCarree(), False),
        ((-200.0, 200.0, 40.0, 60.0), ccrs.NorthPolarStereo(), False),
        # Ending on the seam can frame, and just inside a whole turn a crossing still warns.
        ((-120.0, 180.0, 0.0, 20.0), ccrs.PlateCarree(), False),
        ((1.0, 358.0, 0.0, 20.0), ccrs.PlateCarree(), True),
        # A southern conic spreads across its cut, short of the width it has at the equator.
        ((170.0, 190.0, -70.0, -40.0), ccrs.LambertConformal(central_longitude=0, standard_parallels=(-30, -60), cutoff=30), True),
        # The check cannot measure this map's whole width. It says nothing and the map still draws.
        ((-10.0, 10.0, 40.0, 60.0), ccrs.Orthographic(), False),
    ],
)
def test_levels_extent_warns_when_the_map_cannot_frame_it(caplog, extent, projection, warned):
    _module_log(caplog)
    plot_event_levels(_hand_events(), 0, [0], zdim=ZDIM, extent=extent, projection=projection)
    assert len(_warnings(caplog)) == int(warned)


@pytest.mark.parametrize(
    "extent, projection_type, advised",
    [
        ((170.0, 190.0, 0.0, 20.0), ccrs.PlateCarree, 180.0),
        ((170.0, 190.0, 50.0, 70.0), ccrs.Robinson, 180.0),
        ((180.0, 200.0, 0.0, 20.0), ccrs.Robinson, -170.0),
        ((-100.0, 200.0, 0.0, 20.0), ccrs.PlateCarree, 50.0),
        ((0.0, 360.0, 0.0, 20.0), ccrs.PlateCarree, 180.0),
        ((-20.0, 340.0, 0.0, 20.0), ccrs.PlateCarree, 160.0),
    ],
)
def test_levels_extent_warning_advises_a_centre_that_frames_it(caplog, extent, projection_type, advised):
    _module_log(caplog)
    events = _hand_events()
    plot_event_levels(events, 0, [0], zdim=ZDIM, extent=extent, projection=projection_type())
    message = _warnings(caplog)[0]
    assert "seam" in message and f"central_longitude={advised:g}" in message
    caplog.clear()
    centred = projection_type(central_longitude=advised)
    ax = _map_axes(plot_event_levels(events, 0, [0], zdim=ZDIM, extent=extent, projection=centred))[0]
    assert _warnings(caplog) == []
    if projection_type is ccrs.PlateCarree:
        np.testing.assert_allclose(np.diff(ax.get_xlim()), extent[1] - extent[0], atol=1e-2)


@pytest.mark.parametrize("extent", [(0.0, 720.0, 0.0, 20.0), (-200.0, 200.0, 0.0, 20.0)])
def test_levels_extent_cropped_past_a_whole_turn_suggests_no_centre(caplog, extent):
    _module_log(caplog)
    plot_event_levels(_hand_events(), 0, [0], zdim=ZDIM, extent=extent, projection=ccrs.PlateCarree())
    message = _warnings(caplog)[0]
    assert "whole turn" in message and "central_longitude" not in message


@pytest.mark.parametrize("half_cell", [0.5, 0.125, 1.0 / 24.0, 0.025])
def test_levels_extent_warns_when_a_global_grid_edge_past_the_seam_crops_a_column(caplog, half_cell):
    # lon.min() - dx / 2 to lon.max() + dx / 2 on a grid holding both -180 and 180: the window loses the seam column.
    _module_log(caplog)
    extent = (-180.0 - half_cell, 180.0 + half_cell, 0.0, 20.0)
    plot_event_levels(_hand_events(), 0, [0], zdim=ZDIM, extent=extent, projection=ccrs.PlateCarree())
    assert ["crops" in message for message in _warnings(caplog)] == [True]


def test_levels_extent_past_a_whole_turn_measures_the_crop_against_the_projections_centre(caplog):
    # Against the width centred on Greenwich this window looks whole, though it loses 150W at the equator.
    _module_log(caplog)
    projection = ccrs.Robinson(central_longitude=30)
    plot_event_levels(_hand_events(), 0, [0], zdim=ZDIM, extent=(-200.0, 200.0, -60.0, 60.0), projection=projection)
    assert ["crops" in message for message in _warnings(caplog)] == [True]


def test_levels_extent_past_a_whole_turn_never_advises_a_centre_on_a_polar_map(caplog):
    # drawn is 1.03 of framed here, which is the seam test's width branch on a range under a whole turn.
    _module_log(caplog)
    projection = ccrs.NorthPolarStereo(central_longitude=-170)
    plot_event_levels(_hand_events(), 0, [0], zdim=ZDIM, extent=(-181.0, 181.0, 80.0, 85.0), projection=projection)
    assert not any("seam" in message or "central_longitude" in message for message in _warnings(caplog))


def test_levels_extent_check_frames_at_the_latitudes_of_the_window(caplog):
    # The polar cap frames at any latitude, but its width at 50-70N is not its width at the equator.
    _module_log(caplog)
    plot_event_levels(_hand_events(), 0, [0], zdim=ZDIM, extent=(-180.0, 180.0, 50.0, 70.0), projection=ccrs.NorthPolarStereo())
    assert _warnings(caplog) == []


def test_levels_extent_check_keeps_the_collapse_warning_of_its_throwaway_axes_to_itself(caplog):
    _module_log(caplog)
    with warnings.catch_warnings(record=True) as seen:
        warnings.simplefilter("always")
        vol_mod._warn_if_extent_misdrawn((0.0, 360.0, 0.0, 20.0), ccrs.PlateCarree())
    assert [str(w.message) for w in seen] == []
    assert "collapses" in _warnings(caplog)[0]


def test_levels_extent_warns_once_for_a_figure_of_many_panels(caplog):
    _module_log(caplog)
    plot_event_levels(_hand_events(), 0, [0, 1, 2], zdim=ZDIM, extent=(170.0, 190.0, 0.0, 20.0), projection=ccrs.PlateCarree())
    assert len(_warnings(caplog)) == 1


def _inclusive_global_events(columns):
    """Event 1 on a global 0..360 inclusive grid, where the two edge columns are the same meridian."""
    events = _global_events(range(0))
    lon = np.arange(0.0, 361.0)
    ids = np.zeros((1, 2, events["lat"].size, lon.size), dtype=np.int32)
    ids[0, 0, 100:120, list(columns)] = 1
    coords = {"time": events["time"].values, ZDIM: events[ZDIM].values, "lat": events["lat"].values, "lon": lon}
    return xr.Dataset({"ID_field": ((("time", ZDIM, "lat", "lon")), ids)}, coords={**coords, "ID": [1]})


@pytest.mark.parametrize(
    "columns",
    [[*range(0, 6), *range(355, 361)], range(10, 351)],
    ids=["both_edge_columns", "wide_within_the_grid"],
)
def test_levels_zoom_on_an_inclusive_global_grid_draws_the_whole_width_and_warns_once(caplog, columns):
    _module_log(caplog)
    flat = ccrs.PlateCarree()
    ax = _map_axes(
        plot_event_levels(_inclusive_global_events(columns), 0, [0], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=flat)
    )[0]
    np.testing.assert_allclose(np.diff(ax.get_xlim()), 360.0, atol=1.5)
    messages = _warnings(caplog)
    assert len(messages) == 1 and "seam" in messages[0] and "central_longitude" in messages[0]


def test_levels_zoom_on_a_descending_grid_holds_the_event():
    events = _hand_events().isel(lat=slice(None, None, -1), lon=slice(None, None, -1))
    flat = ccrs.PlateCarree()
    ax = _map_axes(plot_event_levels(events, 0, [0, 1], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=flat))[0]
    np.testing.assert_allclose(ax.get_extent(flat), (-1.0, 25.0, 2.0, 11.0), atol=1e-6)


def test_levels_zoom_across_the_seam_of_a_descending_global_grid_warns_once(caplog):
    _module_log(caplog)
    events = _global_events([*range(0, 6), *range(354, 360)]).isel(lon=slice(None, None, -1))
    plot_event_levels(events, 0, [0], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=ccrs.PlateCarree())
    messages = _warnings(caplog)
    assert len(messages) == 1 and "seam" in messages[0]


def _regional_events(west, east, columns):
    """Event 1 on a regional 1-degree grid from west to east, in the given columns of level 0."""
    events = _global_events(range(0))
    lon = np.arange(west, east + 1.0)
    ids = np.zeros((1, 2, events["lat"].size, lon.size), dtype=np.int32)
    ids[0, 0, 100:120, list(columns)] = 1
    coords = {"time": events["time"].values, ZDIM: events[ZDIM].values, "lat": events["lat"].values, "lon": lon}
    return xr.Dataset({"ID_field": ((("time", ZDIM, "lat", "lon")), ids)}, coords={**coords, "ID": [1]})


def test_levels_zoom_on_a_wide_regional_grid_keeps_its_margin_past_the_edge(caplog):
    _module_log(caplog)
    flat = ccrs.PlateCarree()
    events = _regional_events(-160.0, 160.0, range(0, 3))  # 320 degrees wide, so the grid does not close
    west, _, _, _ = _map_axes(plot_event_levels(events, 0, [0], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=flat))[
        0
    ].get_extent(flat)
    np.testing.assert_allclose(west, -160.0 - 0.15 * 19.0, atol=1e-6)  # the margin is 15% of the 19-degree latitude span
    assert _warnings(caplog) == []


def test_levels_zoom_clamps_a_margin_that_would_pass_a_whole_turn_on_a_regional_grid(caplog):
    _module_log(caplog)
    flat = ccrs.PlateCarree()
    events = _regional_events(-170.0, 170.0, range(0, 340))  # spans 339 degrees; its 15% margin alone passes 360
    west, east, _, _ = _map_axes(plot_event_levels(events, 0, [0], zdim=ZDIM, event_id=1, zoom_to_event=True, projection=flat))[
        0
    ].get_extent(flat)
    np.testing.assert_allclose((west, east), (-170.0, 170.0), atol=1e-6)
    assert _warnings(caplog) == []


def test_levels_on_absent_event_without_zoom_stays_quiet(caplog):
    _module_log(caplog)
    plot_event_levels(_hand_events(), 0, [2], zdim=ZDIM, event_id=1, projection=ccrs.PlateCarree())
    assert _warnings(caplog) == []


def test_levels_each_panel_shows_its_own_level(grow_run):
    events, _, _ = grow_run
    t, levels, ids = _differing_levels(events)
    fig = plot_event_levels(events, t, levels, zdim=ZDIM)
    axes = _map_axes(fig)
    assert len(fig.axes) == len(axes) == len(levels)
    for ax, k in zip(axes, levels):
        shown = _mesh_values(ax).reshape(ids[k].shape)
        expected = np.where(ids[k] > 0, (ids[k] - 1) % 20, np.nan)
        np.testing.assert_array_equal(shown, expected, err_msg=f"panel for level {k}")
        depth = float(events[ZDIM].values[k])
        want = f"{depth:.0f}" if abs(depth) >= 10 else f"{depth:.1f}"
        assert ax.get_title() == f"{ZDIM} = {want} m", f"panel for level {k}"
    assert str(events["time"].values[t])[:10] in fig._suptitle.get_text()


def test_levels_map_is_placed_at_the_dataset_coordinates(grow_run):
    import cartopy.crs as ccrs

    events, _, _ = grow_run
    t, levels, _ = _differing_levels(events)
    fig = plot_event_levels(events, t, levels[:1], zdim=ZDIM, projection=ccrs.PlateCarree())
    mesh = _map_axes(fig)[0].collections[0]
    edges = mesh.get_coordinates()
    lon, lat = events["lon"].values, events["lat"].values
    assert edges.shape == (lat.size + 1, lon.size + 1, 2)
    np.testing.assert_allclose(edges[0, :, 0], np.concatenate([lon - 0.25, [lon[-1] + 0.25]]))
    np.testing.assert_allclose(edges[:, 0, 1], np.concatenate([lat - 0.25, [lat[-1] + 0.25]]))


def test_levels_highlight_marks_only_the_event(merge_run):
    events, _, _ = merge_run
    t, ev = 4, int(events["ID"].values[0])
    ids3 = events["ID_field"].isel(time=t).transpose(ZDIM, "lat", "lon").values
    k = next(k for k in range(ids3.shape[0]) if (ids3[k] == ev).any())
    fig = plot_event_levels(events, t, [k], zdim=ZDIM, event_id=ev)
    shown = _mesh_values(_map_axes(fig)[0])
    ids = ids3[k]
    assert (ids == ev).sum() > 0
    assert (shown == 1.0).sum() == (ids == ev).sum()
    assert (shown == 0.0).sum() == ((ids > 0) & (ids != ev)).sum()


def test_levels_accepts_a_time_label_and_a_projection(merge_run):
    import cartopy.crs as ccrs

    events, _, _ = merge_run
    fig = plot_event_levels(events, events["time"].values[3], [1, 2], zdim=ZDIM, projection=ccrs.PlateCarree())
    assert len(_map_axes(fig)) == 2 and isinstance(_map_axes(fig)[0].projection, ccrs.PlateCarree)


def test_levels_rejects_bad_arguments(merge_run):
    events, _, _ = merge_run
    with pytest.raises(ConfigurationError):
        plot_event_levels(events, 0, [9], zdim=ZDIM)
    with pytest.raises(ConfigurationError):
        plot_event_levels(events, 0, [], zdim=ZDIM)
    with pytest.raises(ConfigurationError):
        plot_event_levels(events, 0, [0], zdim=ZDIM, event_id=999)
    with pytest.raises(ConfigurationError):
        plot_event_levels(events, 0, [0], zdim="not_a_dim")


def test_levels_loads_only_the_requested_time_step(merge_run):
    events, _, _ = merge_run
    lazy, src = _lazy_events(events)
    plot_event_levels(lazy, 5, [0, 2], zdim=ZDIM)
    assert src.times == {5}, f"time steps read: {sorted(src.times)}"


# -- 2. depth-time section ----------------------------------------------------------------------


def _expected_section(events, ev, cell_volume=None):
    """(level, time) quantity of the event over its lifetime, NaN where it does not occupy the level."""
    member = (events["ID_field"] == ev).transpose("time", ZDIM, "lat", "lon")
    if cell_volume is not None:
        member = member * cell_volume
    per = member.sum(["lat", "lon"]).values
    alive = events["presence"].sel(ID=ev).values
    per = per[alive].T.astype(float)
    return np.where(per > 0, per, np.nan)


def test_depth_time_cells_match_the_field(birth_run):
    events, _, _ = birth_run
    ev = int(events["ID"].values[0])
    fig = plot_event_depth_time(events, ev, zdim=ZDIM)
    assert len(fig.axes) == 2  # section + colorbar
    expected = _expected_section(events, ev)
    assert not np.array_equal(expected, expected[::-1], equal_nan=True), "fixture is symmetric in depth"
    drawn = _mesh_values(fig.axes[0]).reshape(expected.shape)
    np.testing.assert_array_equal(drawn, expected)
    # Row j of the section sits at level j's depth, and depth increases downwards.
    y = fig.axes[0].collections[0].get_coordinates()[:, 0, 1]
    d = events[ZDIM].values
    np.testing.assert_allclose(y[1:-1], 0.5 * (d[:-1] + d[1:]))
    assert y[0] < d[0] < y[1] and y[-2] < d[-1] < y[-1]
    assert fig.axes[0].yaxis_inverted()


def test_depth_time_volume_sums_to_the_event_volume(grow_run):
    events, _, cell_volume = grow_run
    ev = int(events["ID"].values[0])
    fig = plot_event_depth_time(events, ev, zdim=ZDIM, quantity="volume", cell_volume=cell_volume)
    expected = _expected_section(events, ev, cell_volume)
    assert not np.array_equal(expected, expected[::-1], equal_nan=True), "fixture is symmetric in depth"
    drawn = _mesh_values(fig.axes[0]).reshape(expected.shape)
    np.testing.assert_allclose(drawn, expected, rtol=1e-12)
    np.testing.assert_allclose(np.nansum(drawn), events["volume"].sel(ID=ev).sum().item(), rtol=1e-12)


def test_depth_time_has_no_offset_text_and_names_the_scale(grow_run):
    events, _, cell_volume = grow_run
    ev = int(events["ID"].values[0])
    big = cell_volume * 1e9  # whole-volume values of order 1e12 or more, as for an ocean event
    fig = plot_event_depth_time(events, ev, zdim=ZDIM, quantity="volume", cell_volume=big)
    fig.canvas.draw()
    colourbar_axes = fig.axes[1]
    assert colourbar_axes.yaxis.get_offset_text().get_text() == ""
    peak = np.nanmax(_mesh_values(fig.axes[0]))
    exponent = int(np.floor(np.log10(peak)))
    assert exponent >= 3
    assert f"10$^{{{exponent}}}$" in colourbar_axes.get_ylabel()
    ticks = [t.get_text() for t in colourbar_axes.get_yticklabels() if t.get_text()]
    assert ticks and all(abs(float(t.replace("\u2212", "-"))) < 10 for t in ticks), (
        f"ticks are not in units of 10^{exponent}: {ticks}"
    )
    assert fig.axes[0].collections[0].get_cmap().name == "magma"
    # Small counts are not rescaled: the colour bar reads as it is, with no power of ten in its label.
    small = plot_event_depth_time(events, ev, zdim=ZDIM)
    assert "10$^" not in small.axes[1].get_ylabel()


def test_depth_time_options(grow_run):
    events, _, _ = grow_run
    ev = int(events["ID"].values[0])
    fig = plot_event_depth_time(events, ev, zdim=ZDIM, cmap="cividis", depth_scale="sqrt")
    assert fig.axes[0].collections[0].get_cmap().name == "cividis"
    assert fig.axes[0].get_yscale() == "function"
    assert fig.axes[0].yaxis_inverted()
    # The square-root scale places 4 m and 16 m at 2 and 4, so thin upper levels get more room.
    np.testing.assert_allclose(fig.axes[0].yaxis.get_transform().transform_non_affine(np.array([4.0, 16.0])), [2.0, 4.0])
    np.testing.assert_allclose(
        plot_event_depth_time(events, ev, zdim=ZDIM).axes[0].yaxis.get_transform().transform_non_affine(np.array([4.0, 16.0])),
        [4.0, 16.0],
    )
    assert plot_event_depth_time(events, ev, zdim=ZDIM).axes[0].get_yscale() == "linear"
    with pytest.raises(ConfigurationError, match="depth_scale"):
        plot_event_depth_time(events, ev, zdim=ZDIM, depth_scale="log")


def test_depth_time_axes_run_from_the_surface_past_the_deepest_level_with_compact_dates(grow_run):
    from matplotlib.dates import ConciseDateFormatter

    events, _, _ = grow_run
    ev = int(events["ID"].values[0])
    depths = events[ZDIM].values
    deepest_edge = float(depths[-1] + (depths[-1] - depths[-2]) / 2)
    for scale in ("linear", "sqrt"):
        ax = plot_event_depth_time(events, ev, zdim=ZDIM, depth_scale=scale).axes[0]
        assert ax.get_ylim() == (deepest_edge, 0.0), scale
        assert isinstance(ax.xaxis.get_major_formatter(), ConciseDateFormatter), scale


def test_depth_time_axis_on_a_crop_starts_at_its_shallowest_level():
    events = _hand_events().assign_coords({ZDIM: [500.0, 700.0, 900.0]})
    for scale in ("linear", "sqrt"):
        ax = plot_event_depth_time(events, 1, zdim=ZDIM, depth_scale=scale).axes[0]
        assert ax.get_ylim() == (1000.0, 400.0), scale


@pytest.mark.parametrize("depths", [[0.49, 1.54, 2.65], [10.0, 20.0, 40.0], [5.0, 50.0, 500.0], [300.0, 500.0, 700.0]])
def test_depth_time_axis_starts_at_the_surface_when_the_top_level_is_near_it(depths):
    events = _hand_events().assign_coords({ZDIM: depths})
    assert plot_event_depth_time(events, 1, zdim=ZDIM).axes[0].get_ylim()[1] == 0.0


@pytest.mark.parametrize(
    "depths, limits",
    [
        ([5.0, 50.0, 500.0], (725.0, 0.0)),
        ([500.0, 50.0, 5.0], (725.0, 0.0)),
        ([-5.0, -50.0, -500.0], (-725.0, 0.0)),
        ([-500.0, -50.0, -5.0], (-725.0, 0.0)),
        ([900.0, 700.0, 500.0], (1000.0, 400.0)),
        ([-500.0, -700.0, -900.0], (-1000.0, -400.0)),
    ],
)
def test_depth_time_axis_keeps_the_surface_up_for_levels_in_any_order_and_sign(depths, limits):
    events = _hand_events().assign_coords({ZDIM: depths})
    assert plot_event_depth_time(events, 1, zdim=ZDIM).axes[0].get_ylim() == limits


def test_depth_time_axis_draws_every_row_of_the_mesh_in_view():
    events = _hand_events().assign_coords({ZDIM: [500.0, 700.0, 900.0]})
    ax = plot_event_depth_time(events, 1, zdim=ZDIM).axes[0]
    y_edges = ax.collections[0].get_coordinates()[:, 0, 1]
    bottom, top = ax.get_ylim()
    assert y_edges.min() >= top and y_edges.max() <= bottom


def test_depth_time_volume_without_cell_volume_raises(grow_run):
    events, _, _ = grow_run
    with pytest.raises(ConfigurationError, match="cell volumes"):
        plot_event_depth_time(events, int(events["ID"].values[0]), zdim=ZDIM, quantity="volume")
    with pytest.raises(ConfigurationError):
        plot_event_depth_time(events, int(events["ID"].values[0]), zdim=ZDIM, quantity="mass")


def test_depth_time_reads_only_the_event_lifetime(grow_run):
    events, _, _ = grow_run
    ev = int(events["ID"].values[0])
    alive = set(np.flatnonzero(events["presence"].sel(ID=ev).values).tolist())
    assert 0 < len(alive) < events.sizes["time"]
    lazy, src = _lazy_events(events)
    plot_event_depth_time(lazy, ev, zdim=ZDIM)
    assert src.times == alive, f"time steps read: {sorted(src.times)}, lifetime: {sorted(alive)}"


# -- 3. event summary ---------------------------------------------------------------------------


def _present(events, ev, name):
    return events[name].sel(ID=ev).where(events["presence"].sel(ID=ev)).values


def _line(ax, i):
    return np.asarray(ax.lines[i].get_ydata(), dtype=float)


def test_summary_series_match_the_dataset(grow_run):
    events, _, _ = grow_run
    ev = int(events["ID"].values[0])
    fig = plot_event_summary(events, ev)
    assert len(fig.axes) == 3
    ax_v, ax_c, ax_e = fig.axes
    np.testing.assert_array_equal(_line(ax_v, 0), _present(events, ev, "volume"))
    np.testing.assert_array_equal(_line(ax_c, 0), _present(events, ev, "depth_centroid"))
    top, bottom = _present(events, ev, "depth_top"), _present(events, ev, "depth_bottom")
    assert np.nanmax(bottom - top) > 0 and not np.array_equal(top, bottom, equal_nan=True)
    np.testing.assert_array_equal(_line(ax_e, 0), top)
    np.testing.assert_array_equal(_line(ax_e, 1), bottom)
    # The shaded extent spans from the top line to the bottom line.
    verts = np.concatenate([path.vertices for path in ax_e.collections[0].get_paths()])
    assert np.isin(top[~np.isnan(top)], verts[:, 1]).all() and np.isin(bottom[~np.isnan(bottom)], verts[:, 1]).all()
    assert ax_e.yaxis_inverted() and ax_c.yaxis_inverted()


def test_summary_uses_compact_date_labels(grow_run):
    from matplotlib.dates import ConciseDateFormatter

    events, _, _ = grow_run
    fig = plot_event_summary(events, int(events["ID"].values[0]))
    assert isinstance(fig.axes[2].xaxis.get_major_formatter(), ConciseDateFormatter)


def test_summary_volume_axis_has_no_offset_text(grow_run):
    events, _, _ = grow_run
    ev = int(events["ID"].values[0])
    big = events.assign(volume=events["volume"] * 1e12)  # volumes of order 1e13 m^3 and more, as for an ocean event
    fig = plot_event_summary(big, ev)
    fig.canvas.draw()
    ax_v = fig.axes[0]
    assert ax_v.yaxis.get_offset_text().get_text() == ""
    exponent = int(np.floor(np.log10(np.nanmax(big["volume"].sel(ID=ev).values))))
    assert f"10$^{{{exponent}}}$" in ax_v.get_ylabel()
    ticks = [t.get_text() for t in ax_v.get_yticklabels() if t.get_text()]
    assert ticks and all(abs(float(t.replace("\u2212", "-"))) < 10 for t in ticks), ticks


def test_summary_marks_this_events_merges_only(merge_run):
    events, merges, _ = merge_run
    ev = int(events["ID"].values[0])
    rows = events["merge_ledger"].sel(ID=ev).values
    active = (rows >= 0).any(axis=1)
    episodes = int((active & ~np.concatenate([[False], active[:-1]])).sum())
    assert episodes == 1
    first_active = events["time"].values[np.flatnonzero(active)[0]]

    def marks(fig):
        return [ln for ln in fig.axes[0].lines[1:] if len(set(ln.get_xdata())) == 1]

    for fig in (plot_event_summary(events, ev), plot_event_summary(events, ev, merges_ds=merges)):
        assert len(marks(fig)) == episodes and marks(fig)[0].get_label() == "merge"
        assert np.datetime64(marks(fig)[0].get_xdata()[0], "ns") == first_active
    # An event the ledger does not list gets no mark, whatever other events do at that time.
    quiet = events.assign(merge_ledger=events["merge_ledger"].where(events["ID"] == ev, -1))
    other = int(events["ID"].values[1])
    assert len(marks(plot_event_summary(quiet, other))) == 0


def test_summary_marks_splits_from_the_merges_dataset(merge_run):
    events, merges, _ = merge_run
    ev = int(events["ID"].values[0])
    days = events["time"].values[[1, 2, 5]]
    with_splits = merges.drop_dims("split_ID", errors="ignore").assign(
        split_time=("split_ID", days),
        split_event_IDs=("split_ID", np.array([ev, ev, ev], dtype=np.int32)),
    )

    def marks(fig, label):
        return [ln for ln in fig.axes[0].lines[1:] if len(set(ln.get_xdata())) == 1 and ln.get_linestyle() == ":" and label]

    fig = plot_event_summary(events, ev, with_splits)
    drawn = [np.datetime64(ln.get_xdata()[0], "ns") for ln in marks(fig, True)]
    # Days 1 and 2 are one run, marked where it begins; day 5 starts another.
    assert drawn == [days[0], days[2]]
    assert "split" in fig.axes[0].get_legend_handles_labels()[1]
    # Without the merges dataset, or for another event, there is no split mark.
    assert marks(plot_event_summary(events, ev), True) == []
    assert marks(plot_event_summary(events, int(events["ID"].values[1]), with_splits), True) == []
    # A merges dataset written before split records existed still draws.
    old = with_splits.drop_vars(["split_time", "split_event_IDs"])
    assert marks(plot_event_summary(events, ev, old), True) == []


def test_summary_marks_a_split_on_a_day_the_event_has_no_cells(merge_run):
    # A merge at the split's timestep can take the children into another event, so the row's own event can be absent.
    # The state is built by hand: it has been seen on 3-D layouts only, and no 4-D producer has been found.
    events, merges, _ = merge_run
    ev = int(events["ID"].values[0])
    day = events["time"].values[-1]
    presence = events["presence"].copy(deep=True).load()
    presence.loc[{"ID": ev, "time": day}] = False
    events = events.assign(presence=presence)
    with_row = merges.drop_dims("split_ID", errors="ignore").assign(
        split_time=("split_ID", np.array([day])),
        split_event_IDs=("split_ID", np.array([ev], dtype=np.int32)),
    )
    fig = plot_event_summary(events, ev, with_row)
    drawn = [
        np.datetime64(ln.get_xdata()[0], "ns")
        for ln in fig.axes[0].lines[1:]
        if len(set(ln.get_xdata())) == 1 and ln.get_linestyle() == ":"
    ]
    assert drawn == [day]


def test_summary_without_a_merge_has_no_marks(grow_run):
    events, _, _ = grow_run
    fig = plot_event_summary(events, int(events["ID"].values[0]))
    assert len(fig.axes[0].lines) == 1


# -- time axes -----------------------------------------------------------------------------------


def _retimed(events, times):
    return events.assign_coords(time=times)


def test_non_daily_time_axes(grow_run):
    import cftime

    events, _, _ = grow_run
    nt = events.sizes["time"]
    ev = int(events["ID"].values[0])
    t, levels, _ = _differing_levels(events)

    hourly = _retimed(events, np.datetime64("2000-01-01T00:00") + 6 * np.arange(nt).astype("timedelta64[h]"))
    fig = plot_event_levels(hourly, t, levels[:1], zdim=ZDIM)
    assert "06:00" in fig._suptitle.get_text() or "12:00" in fig._suptitle.get_text() or "18:00" in fig._suptitle.get_text()
    assert str(hourly["time"].values[t])[:13].replace("T", " ") in fig._suptitle.get_text()
    plot_event_summary(hourly, ev)

    noleap = _retimed(events, [cftime.DatetimeNoLeap(2001, 1, 1 + i) for i in range(nt)])
    fig = plot_event_levels(noleap, t, levels[:1], zdim=ZDIM)
    assert "2001-01" in fig._suptitle.get_text()
    fig = plot_event_summary(noleap, ev)
    assert "days since" in fig.axes[2].get_xlabel()
    plot_event_depth_time(noleap, ev, zdim=ZDIM)

    steps = _retimed(events, np.arange(nt))
    plot_event_levels(steps, t, levels[:1], zdim=ZDIM)
    plot_event_summary(steps, ev)
    plot_event_depth_time(steps, ev, zdim=ZDIM)


# -- 4. animation -------------------------------------------------------------------------------


@pytest.fixture
def stub_encoder(monkeypatch, tmp_path):
    """Replace ffmpeg only: record every frame's pixel size and write the output file."""
    monkeypatch.setattr(animation_mod.shutil, "which", lambda _: "/usr/bin/ffmpeg")
    seen = {"sizes": [], "names": []}

    def fake_run(cmd, *args, **kwargs):
        frames = sorted(Path(cmd[cmd.index("-i") + 1]).parent.glob("time_*.jpg"))
        seen["names"] = [p.name for p in frames]
        seen["sizes"] = [Image.open(p).size for p in frames]
        Path(cmd[-1]).write_bytes(b"movie")

    monkeypatch.setattr(animation_mod.subprocess, "run", fake_run)
    sizing_calls = []
    real = animation_mod._domain_figsize

    def spy(*a, **k):
        sizing_calls.append(1)
        return real(*a, **k)

    monkeypatch.setattr(animation_mod, "_domain_figsize", spy)
    with dask.config.set({"temporary-directory": str(tmp_path)}):
        yield seen, sizing_calls


def _graticule_labels(ax, gridliner_type):
    """The visible labels of a map's graticule on all four sides: a curved map does not label only the bottom edge."""
    return [
        label
        for g in ax.artists
        if isinstance(g, gridliner_type)
        for side in (g.bottom_label_artists, g.top_label_artists, g.left_label_artists, g.right_label_artists)
        for label in side
        if label.get_visible()
    ]


@pytest.fixture
def frames(monkeypatch):
    """Read every animation frame's drawn artists back off the figure as it is saved."""
    from cartopy.mpl.gridliner import Gridliner
    from matplotlib.figure import Figure as RealFigure

    recorded = []

    class Recording(RealFigure):
        def savefig(self, *args, **kwargs):
            super().savefig(*args, **kwargs)
            maps = [a for a in self.axes if hasattr(a, "projection")]
            others = [a for a in self.axes if not hasattr(a, "projection")]
            recorded.append(
                {
                    "extents": [a.get_extent(ccrs.PlateCarree()) for a in maps],
                    "titles": [a.get_title() for a in maps],
                    "title_colours": [a.title.get_color() for a in maps if a.get_title()],
                    "suptitle_colour": self._suptitle.get_color(),
                    "bar_label_colours": [a.xaxis.label.get_color() for a in others],
                    "tick_labels": [[t.get_text() for t in _graticule_labels(a, Gridliner)] for a in maps],
                    "tick_label_colours": [[t.get_color() for t in _graticule_labels(a, Gridliner)] for a in maps],
                }
            )

    monkeypatch.setattr(vol_mod, "Figure", Recording)
    return recorded


def _luminance(colour):
    from matplotlib.colors import to_rgb

    r, g, b = to_rgb(colour)
    return 0.2126 * r + 0.7152 * g + 0.0722 * b


def _window(events):
    lons, lats = events["lon"].values, events["lat"].values
    span = lons[-1] - lons[0]
    return (float(lons[0]), float(lons[0] + 0.25 * span), float(lats[0]), float(lats[-1]))


@pytest.mark.parametrize("levels", ["footprint", [0, 1]])
def test_animation_extent_zooms_every_map(merge_run, stub_encoder, frames, tmp_path, levels):
    events, _, _ = merge_run
    window = _window(events)
    flat = PlotConfig(projection=ccrs.PlateCarree())
    animate_event_levels(events, zdim=ZDIM, levels=levels, plot_dir=str(tmp_path), extent=window, config=flat)
    assert len(frames) == events.sizes["time"]
    for frame in frames:
        for drawn in frame["extents"]:
            np.testing.assert_allclose(drawn, window, atol=1e-6)
    frames.clear()
    animate_event_levels(events, zdim=ZDIM, levels=levels, plot_dir=str(tmp_path), config=flat)
    assert not np.allclose(frames[0]["extents"][0], window), "the whole domain is drawn without an extent"


@pytest.mark.parametrize("levels", ["footprint", [0]])
def test_animation_extent_across_the_projection_seam_warns_once(merge_run, stub_encoder, tmp_path, caplog, levels):
    _module_log(caplog)
    events, _, _ = merge_run
    animate_event_levels(
        events, zdim=ZDIM, levels=levels, plot_dir=str(tmp_path), file_name="seam", extent=(170.0, 190.0, 0.0, 20.0)
    )
    assert sum("seam" in message for message in _warnings(caplog)) == 1


def test_animation_extent_check_uses_the_animations_projection(merge_run, stub_encoder, tmp_path, caplog):
    _module_log(caplog)
    events, _, _ = merge_run
    centred = PlotConfig(projection=ccrs.PlateCarree(central_longitude=180))
    animate_event_levels(
        events, zdim=ZDIM, levels=[0], plot_dir=str(tmp_path), file_name="centred", extent=(170.0, 190.0, 0.0, 20.0), config=centred
    )
    assert not any("seam" in message for message in _warnings(caplog))


def test_animation_level_panels_size_the_canvas_to_the_window(merge_run, stub_encoder, tmp_path):
    events, _, _ = merge_run
    seen, _ = stub_encoder
    animate_event_levels(events, zdim=ZDIM, levels=[0, 1], plot_dir=str(tmp_path), file_name="whole")
    (whole,) = set(seen["sizes"])
    animate_event_levels(events, zdim=ZDIM, levels=[0, 1], plot_dir=str(tmp_path), file_name="zoomed", extent=_window(events))
    (zoomed,) = set(seen["sizes"])
    assert zoomed[0] / zoomed[1] < whole[0] / whole[1]


def test_animation_level_panel_titles_round_the_depth(merge_run, stub_encoder, frames, tmp_path):
    events, _, _ = merge_run
    depths = 0.494 + 13.3 * np.arange(events.sizes[ZDIM], dtype=float)
    animate_event_levels(events.assign_coords({ZDIM: depths}), zdim=ZDIM, levels=[0, 1], plot_dir=str(tmp_path))
    assert frames[0]["titles"] == [f"{ZDIM} = 0.5 m", f"{ZDIM} = 14 m"]


def test_animation_dark_theme_colours_the_text_light_and_light_theme_dark(merge_run, stub_encoder, frames, tmp_path):
    events, _, _ = merge_run
    for theme, is_light in (("dark", True), ("light", False)):
        for levels in ("footprint", [0, 1]):
            frames.clear()
            animate_event_levels(events, zdim=ZDIM, levels=levels, plot_dir=str(tmp_path), theme=theme)
            frame = frames[0]
            texts = [frame["suptitle_colour"], *frame["title_colours"], *frame["bar_label_colours"]]
            assert len(texts) >= 2
            for colour in texts:
                assert (_luminance(colour) > 0.6) == is_light, (theme, levels, colour)


@pytest.mark.parametrize("projection", [ccrs.PlateCarree(), ccrs.Robinson()], ids=["PlateCarree", "Robinson"])
def test_animation_footprint_draws_graticule_labels_only_when_asked(merge_run, stub_encoder, frames, tmp_path, projection):
    events, _, _ = merge_run
    options = dict(projection=projection, grid_lines=True)
    animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), config=PlotConfig(grid_labels=True, **options))
    assert frames[0]["tick_labels"][0], "the graticule is labelled"
    frames.clear()
    animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), config=PlotConfig(**options))
    assert frames[0]["tick_labels"][0] == []


@pytest.mark.parametrize("projection", [ccrs.PlateCarree(), ccrs.Robinson()], ids=["PlateCarree", "Robinson"])
def test_animation_graticule_labels_take_the_colour_of_the_theme(merge_run, stub_encoder, frames, tmp_path, projection):
    events, _, _ = merge_run
    config = PlotConfig(grid_labels=True, grid_lines=True, projection=projection)
    animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), config=config, theme="dark")
    dark = frames[0]["tick_label_colours"][0]
    assert dark and all(_luminance(colour) > 0.5 for colour in dark)
    frames.clear()
    animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), config=config)
    light = frames[0]["tick_label_colours"][0]
    assert light and all(_luminance(colour) < 0.5 for colour in light)


def test_animation_frames_share_one_size(merge_run, stub_encoder, tmp_path):
    events, _, _ = merge_run
    seen, sizing_calls = stub_encoder
    out = animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), file_name="fp")
    assert out == str(tmp_path / "fp.mp4") and Path(out).read_bytes() == b"movie"
    assert seen["names"] == [f"time_{i:04d}.jpg" for i in range(events.sizes["time"])]
    assert len(set(seen["sizes"])) == 1, f"frame sizes differ: {sorted(set(seen['sizes']))}"
    assert len(sizing_calls) == 1, "the canvas must be sized once per movie, not per frame"


def test_animation_colours_the_shallowest_occupied_depth(merge_run, stub_encoder, monkeypatch, tmp_path):
    events, _, _ = merge_run
    drawn = []
    real = vol_mod.pcolormesh_seam_safe

    def spy_mesh(ax, lons, lats, values, **kw):
        drawn.append((np.array(values), kw["norm"]))
        return real(ax, lons, lats, values, **kw)

    monkeypatch.setattr(vol_mod, "pcolormesh_seam_safe", spy_mesh)
    ids = events["ID_field"].transpose("time", ZDIM, "lat", "lon").values
    depth = events[ZDIM].values
    expected = np.full((ids.shape[0],) + ids.shape[2:], np.nan)
    for k in range(ids.shape[1] - 1, -1, -1):  # deepest first, so the shallowest wins
        expected = np.where(ids[:, k] > 0, depth[k], expected)
    animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path))
    assert len(drawn) == ids.shape[0], "one map per time step"
    for t, (values, _) in enumerate(drawn):
        np.testing.assert_array_equal(values, expected[t], err_msg=f"frame {t}")
    assert len({v.tobytes() for v, _ in drawn}) > 1, "fixture frames are too alike to tell time steps apart"
    assert (drawn[0][1].vmin, drawn[0][1].vmax) == (depth.min(), depth.max())


def test_animation_extent_sizes_the_canvas_to_the_window(merge_run, stub_encoder, tmp_path):
    events, _, _ = merge_run
    seen, _ = stub_encoder
    lons, lats = events["lon"].values, events["lat"].values
    animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), file_name="whole")
    (whole_size,) = set(seen["sizes"])
    # A window twice as tall as it is wide, in a domain that is not: the canvas must follow the window.
    span = lons[-1] - lons[0]
    window = (lons[0], lons[0] + 0.25 * span, lats[0], lats[-1])
    animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), file_name="zoomed", extent=window)
    (zoomed_size,) = set(seen["sizes"])
    assert all(side % 2 == 0 for side in zoomed_size + whole_size), "h264 needs even pixel counts"
    assert zoomed_size[0] / zoomed_size[1] < whole_size[0] / whole_size[1]
    with pytest.raises(ConfigurationError):
        animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), extent=(1, 2, 3))


def test_animation_dark_theme_draws_a_dark_map(merge_run, stub_encoder, monkeypatch, tmp_path):
    events, _, _ = merge_run
    seen = {}
    real = vol_mod._add_map_features

    def spy(ax, grid_lines, extent=None, theme="light", **kw):
        real(ax, grid_lines, extent, theme, **kw)
        seen.setdefault(theme, []).append((ax.get_facecolor()[:3], ax.figure.get_facecolor()[:3]))

    monkeypatch.setattr(vol_mod, "_add_map_features", spy)
    for theme in ("light", "dark"):
        animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), theme=theme)
        animate_event_levels(events, zdim=ZDIM, levels=[0, 1], plot_dir=str(tmp_path), theme=theme)
    assert set(seen) == {"light", "dark"}
    for ocean, background in seen["dark"]:
        assert max(ocean) < 0.4 and max(background) < 0.2
    for ocean, background in seen["light"]:
        assert min(ocean) > 0.8 and min(background) > 0.9
    with pytest.raises(ConfigurationError, match="theme"):
        animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), theme="neon")


def test_animation_footprint_honours_the_graticule_option_and_has_a_coloured_ocean(merge_run, stub_encoder, monkeypatch, tmp_path):
    events, _, _ = merge_run
    calls = {}
    real = vol_mod._add_map_features

    def spy(ax, grid_lines, extent=None, theme="light", **kw):
        calls.setdefault("grid", []).append(grid_lines)
        real(ax, grid_lines, extent, theme, **kw)
        calls["ocean"] = ax.get_facecolor()

    monkeypatch.setattr(vol_mod, "_add_map_features", spy)
    animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), config=PlotConfig(grid_lines=False))
    assert set(calls["grid"]) == {False}
    assert calls["ocean"][:3] != (1.0, 1.0, 1.0), "the ocean must not be left white"


def test_animation_level_panels_share_one_size(merge_run, stub_encoder, monkeypatch, tmp_path):
    events, _, _ = merge_run
    seen, sizing_calls = stub_encoder
    levels = [1, 2, 3]
    drawn = []
    real = vol_mod.pcolormesh_seam_safe

    def spy_mesh(ax, lons, lats, values, **kw):
        drawn.append(np.array(values))
        return real(ax, lons, lats, values, **kw)

    monkeypatch.setattr(vol_mod, "pcolormesh_seam_safe", spy_mesh)
    out = animate_event_levels(events, zdim=ZDIM, levels=levels, plot_dir=str(tmp_path), file_name="panels")
    nt = events.sizes["time"]
    assert out == str(tmp_path / "panels.mp4") and Path(out).read_bytes() == b"movie"
    assert seen["names"] == [f"time_{i:04d}.jpg" for i in range(nt)]
    assert len(set(seen["sizes"])) == 1, f"frame sizes differ: {sorted(set(seen['sizes']))}"
    assert all(w % 2 == 0 and h % 2 == 0 for w, h in seen["sizes"])
    assert len(drawn) == nt * len(levels), "one panel per level per frame"
    # Panel j of frame t shows level levels[j] at time step t (frames in time order, panels in level order).
    ids = events["ID_field"].transpose("time", ZDIM, "lat", "lon").values
    expected = [np.where(ids[t, k] > 0, (ids[t, k] - 1) % 20, np.nan) for t in range(nt) for k in levels]
    assert len({e.tobytes() for e in expected}) > len(levels), "fixture frames are too alike to tell panels or steps apart"
    for n, (got, want) in enumerate(zip(drawn, expected)):
        np.testing.assert_array_equal(got, want, err_msg=f"frame {n // len(levels)}, panel {n % len(levels)}")
    assert len(sizing_calls) == 1, "the canvas must be sized once per movie, not per frame"


def test_animation_level_panels_without_ffmpeg_returns_none(merge_run, monkeypatch, tmp_path):
    events, _, _ = merge_run
    monkeypatch.setattr(animation_mod.shutil, "which", lambda _: None)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert animate_event_levels(events, zdim=ZDIM, levels=[0, 1], plot_dir=str(tmp_path)) is None


def test_animation_rejects_bad_levels(merge_run, tmp_path):
    events, _, _ = merge_run
    for bad in ("everything", [], [7]):
        with pytest.raises(ConfigurationError):
            animate_event_levels(events, zdim=ZDIM, levels=bad, plot_dir=str(tmp_path))


def test_animation_without_ffmpeg_returns_none(merge_run, monkeypatch, tmp_path):
    events, _, _ = merge_run
    monkeypatch.setattr(animation_mod.shutil, "which", lambda _: None)
    with warnings.catch_warnings(record=True):
        warnings.simplefilter("always")
        assert animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path)) is None


@pytest.mark.skipif(shutil.which("ffmpeg") is None, reason="ffmpeg not installed")
def test_animation_writes_a_movie(merge_run, tmp_path):
    events, _, _ = merge_run
    out = animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), file_name="movie", config=PlotConfig(framerate=2))
    # The call returns after the encoder exits, so the file is complete here; no waiting on time.
    assert out is not None and Path(out).is_file() and Path(out).stat().st_size > 0
