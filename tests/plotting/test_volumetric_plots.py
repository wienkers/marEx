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


def test_levels_title_prints_depth_to_four_significant_figures(grow_run):
    events, _, _ = grow_run
    t, levels, _ = _differing_levels(events)
    long_floats = events.assign_coords({ZDIM: 1234.5678 + np.arange(events.sizes[ZDIM], dtype=float)})
    fig = plot_event_levels(long_floats, t, levels[:1], zdim=ZDIM)
    expected = f"{ZDIM} = {1234.5678 + levels[0]:.4g} m (level {levels[0]})"
    assert _map_axes(fig)[0].get_title() == expected


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
        assert f"level {k}" in ax.get_title()
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


def test_animation_frames_share_one_size(merge_run, stub_encoder, tmp_path):
    events, _, _ = merge_run
    seen, sizing_calls = stub_encoder
    out = animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path), file_name="fp")
    assert out == str(tmp_path / "fp.mp4") and Path(out).read_bytes() == b"movie"
    assert seen["names"] == [f"time_{i:04d}.jpg" for i in range(events.sizes["time"])]
    assert len(set(seen["sizes"])) == 1, f"frame sizes differ: {sorted(set(seen['sizes']))}"
    assert len(sizing_calls) == 1, "the canvas must be sized once per movie, not per frame"


def test_animation_colours_the_shallowest_occupied_depth(merge_run, monkeypatch, tmp_path):
    events, _, _ = merge_run
    captured = {}
    monkeypatch.setattr(vol_mod, "_animate", lambda plotter, config, **kw: captured.update(da=plotter.da, config=config) or "x")
    ids = events["ID_field"].transpose("time", ZDIM, "lat", "lon").values
    depth = events[ZDIM].values
    expected = np.full((ids.shape[0],) + ids.shape[2:], np.nan)
    for k in range(ids.shape[1] - 1, -1, -1):  # deepest first, so the shallowest wins
        expected = np.where(ids[:, k] > 0, depth[k], expected)
    animate_event_levels(events, zdim=ZDIM, plot_dir=str(tmp_path))
    np.testing.assert_array_equal(captured["da"].transpose("time", "lat", "lon").values, expected)
    assert captured["config"].clim == (depth.min(), depth.max())


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
