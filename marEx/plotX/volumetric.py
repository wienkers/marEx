"""
Plotting helpers for 4-D ``(time, depth, lat, lon)`` event output.

Four helpers draw the events dataset a 4-D tracker returns:

* :func:`plot_event_levels`: map panels of the event IDs at selected depth levels for one time step
* :func:`plot_event_depth_time`: one event's occupied cell count or volume per level against time
* :func:`plot_event_summary`: one event's volume, depth centroid and depth extent against time, with
  merges and splits marked
* :func:`animate_event_levels`: the depth-projected footprint coloured by shallowest occupied depth

Every helper selects the time steps, levels or event it needs from the lazy ID field before loading
anything, so a field larger than memory is never materialised to draw one panel.
"""

import gc
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from ..exceptions import ConfigurationError
from ..logging_config import get_logger
from . import animation as _anim
from .animation import _animate
from .config import PlotConfig
from .dependencies import _check_plotting_dependencies
from .gridded import GriddedPlotter
from .mesh import pcolormesh_seam_safe

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.figure import Figure
except ImportError:  # pragma: no cover - checked by _check_plotting_dependencies at call time
    ccrs = None
    cfeature = None
    plt = None
    ListedColormap = None
    Normalize = None
    Figure = None

try:
    from typing import Literal
except ImportError:  # pragma: no cover
    from typing_extensions import Literal  # type: ignore

logger = get_logger(__name__)

__all__ = ["plot_event_levels", "plot_event_depth_time", "plot_event_summary", "animate_event_levels"]

_ID_FIELD = "ID_field"
_TIME = "time"
# Event IDs are drawn from a cycling palette of this many colours, so the number of IDs (which
# grows with the run) never sets the size of the colour map.
_N_ID_COLOURS = 20
_PANEL_COLUMNS = 3


# --------------------------------------------------------------------------------------
# Argument handling
# --------------------------------------------------------------------------------------


def _horizontal_dims(id_field: xr.DataArray, zdim: str) -> Tuple[str, str]:
    """Return the ``(y, x)`` dimension names of the ID field: the two dimensions that are neither time nor depth."""
    if _TIME not in id_field.dims or zdim not in id_field.dims:
        raise ConfigurationError(
            f"The ID field has dimensions {tuple(id_field.dims)}, expected '{_TIME}' and '{zdim}' among them",
            suggestions=["Pass zdim as the name of the depth dimension of the events dataset"],
            context={"dims": tuple(id_field.dims), "zdim": zdim},
        )
    rest = [str(d) for d in id_field.dims if d not in (_TIME, zdim)]
    if len(rest) != 2:
        raise ConfigurationError(
            f"Expected two horizontal dimensions besides '{_TIME}' and '{zdim}', found {rest}",
            context={"dims": tuple(id_field.dims)},
        )
    if "lat" in rest and "lon" in rest:
        return "lat", "lon"
    return rest[0], rest[1]


def _id_field(events_ds: xr.Dataset, zdim: str) -> xr.DataArray:
    if _ID_FIELD not in events_ds:
        raise ConfigurationError(
            f"The events dataset has no '{_ID_FIELD}' variable",
            suggestions=["Pass the events dataset returned by the 4-D tracker"],
        )
    return events_ds[_ID_FIELD]


def _check_event(events_ds: xr.Dataset, event_id: int) -> int:
    """Validate that ``event_id`` is an event of the dataset and return it as an int."""
    if "ID" not in events_ds.coords or int(event_id) not in set(events_ds["ID"].values.tolist()):
        raise ConfigurationError(
            f"Event {event_id} is not in the events dataset",
            suggestions=["Choose an event ID from events_ds['ID']"],
            context={"event_id": event_id},
        )
    return int(event_id)


def _depth_values(id_field: xr.DataArray, zdim: str) -> np.ndarray:
    """Depth of each level (positive down, metres); the level index when the dimension has no coordinate."""
    if zdim in id_field.coords:
        return np.asarray(id_field[zdim].values, dtype=np.float64)
    return np.arange(id_field.sizes[zdim], dtype=np.float64)


def _select_time(id_field: xr.DataArray, time: Any) -> xr.DataArray:
    """One time step by integer index or by (nearest) time label, still lazy."""
    if isinstance(time, (int, np.integer)):
        return id_field.isel({_TIME: int(time)})
    return id_field.sel({_TIME: time}, method="nearest")


def _time_text(value: Any) -> str:
    """Return a label for one time value that keeps the time of day when it is not midnight."""
    if isinstance(value, np.datetime64):
        text = str(np.datetime_as_string(value, unit="s"))
        if text.endswith("T00:00:00"):
            return text[:10]
        return text.replace("T", " ")[:16] if text.endswith(":00") else text.replace("T", " ")
    return str(value)


def _time_axis(times: np.ndarray) -> Tuple[np.ndarray, str]:
    """Return plottable x values and an axis label for a time coordinate of any calendar."""
    if times.dtype.kind in "Mifu":
        return times, _TIME
    # Non-standard calendars (cftime) have no matplotlib date converter: use elapsed days.
    days = np.array([(t - times[0]).total_seconds() / 86400.0 for t in times])
    return days, f"days since {times[0]}"


def _id_style(event_id: Optional[int]) -> Tuple[Any, Any]:
    """Return the colour map and norm for drawing IDs, or one highlighted event over the rest."""
    if event_id is None:
        return plt.get_cmap("tab20"), Normalize(-0.5, _N_ID_COLOURS - 0.5)
    return ListedColormap(["lightgrey", "tab:red"]), Normalize(-0.5, 1.5)


def _id_values(level_ids: np.ndarray, event_id: Optional[int]) -> np.ndarray:
    """Return the colour index drawn for each cell of one level (NaN where no event)."""
    if event_id is None:
        return np.where(level_ids > 0, (level_ids - 1) % _N_ID_COLOURS, np.nan).astype(float)
    return np.where(level_ids == event_id, 1.0, np.where(level_ids > 0, 0.0, np.nan))


def _add_map_features(ax: Any, grid_lines: bool) -> None:
    """Land, coastlines and optional graticule, as the other plotX maps draw them."""
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor="darkgrey", zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.5, zorder=3)
    if grid_lines:
        ax.gridlines(crs=ccrs.PlateCarree(), draw_labels=False, linewidth=1, color="gray", alpha=0.5, linestyle="--", zorder=4)


# --------------------------------------------------------------------------------------
# 1. Per-level map panels
# --------------------------------------------------------------------------------------


def plot_event_levels(
    events_ds: xr.Dataset,
    time: Any,
    levels: Sequence[int],
    *,
    zdim: str,
    event_id: Optional[int] = None,
    config: Optional[PlotConfig] = None,
    projection: Optional[Any] = None,
) -> "Figure":
    """
    Map panels of the event IDs at selected depth levels for one time step.

    Parameters
    ----------
    events_ds : xarray.Dataset
        4-D events dataset with an ``ID_field`` of dimensions ``(time, zdim, y, x)``.
    time : int or datetime-like
        The time step: an integer index, or a time label (the nearest step is used).
    levels : sequence of int
        Indices along ``zdim`` of the levels to draw, one panel each.
    zdim : str
        Name of the depth dimension.
    event_id : int, optional
        Highlight this event in red over the other events in grey. By default every event is drawn,
        each in its own colour.
    config : PlotConfig, optional
        Supplies the projection and whether grid lines are drawn.
    projection : cartopy.crs.CRS, optional
        Overrides the projection of ``config`` (default Robinson).

    Returns
    -------
    matplotlib.figure.Figure
        One map axes per level.
    """
    _check_plotting_dependencies()
    config = config or PlotConfig()
    ids = _id_field(events_ds, zdim)
    ydim, xdim = _horizontal_dims(ids, zdim)
    levels = [int(k) for k in levels]
    if not levels:
        raise ConfigurationError("levels is empty", suggestions=["Pass at least one level index"])
    nz = ids.sizes[zdim]
    bad = [k for k in levels if not -nz <= k < nz]
    if bad:
        raise ConfigurationError(
            f"Level indices {bad} are outside the {nz} levels of '{zdim}'",
            context={"levels": levels, "n_levels": nz},
        )
    if event_id is not None:
        event_id = _check_event(events_ds, event_id)

    # Select the time step and levels first; only this small block is loaded.
    block = _select_time(ids, time).isel({zdim: levels}).transpose(zdim, ydim, xdim).load()
    depths = _depth_values(ids, zdim)[levels]
    time_label = _time_text(block[_TIME].values[()]) if _TIME in block.coords else ""

    proj = projection if projection is not None else config.projection
    proj = proj if proj is not None else ccrs.Robinson()
    n = len(levels)
    ncols = min(_PANEL_COLUMNS, n)
    nrows = int(np.ceil(n / ncols))
    fig = plt.figure(figsize=(4.2 * ncols, 2.6 * nrows + 0.4))
    axes = fig.subplots(nrows, ncols, squeeze=False, subplot_kw={"projection": proj}).flatten()

    cmap, norm = _id_style(event_id)

    lons = block[xdim].values
    lats = block[ydim].values
    for i, ax in enumerate(axes):
        if i >= n:
            fig.delaxes(ax)
            continue
        level_ids = block.isel({zdim: i}).values
        values = _id_values(level_ids, event_id)
        pcolormesh_seam_safe(ax, lons, lats, values, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto")
        _add_map_features(ax, config.grid_lines)
        ax.set_title(f"{zdim} = {depths[i]:.4g} m (level {levels[i]})", size=10)

    heading = config.title or (f"Event {event_id}" if event_id is not None else "Event IDs")
    fig.suptitle(f"{heading}, {time_label}" if time_label else heading, size=12)
    return fig


# --------------------------------------------------------------------------------------
# 2. Depth-time section
# --------------------------------------------------------------------------------------


def plot_event_depth_time(
    events_ds: xr.Dataset,
    event_id: int,
    *,
    zdim: str,
    quantity: "Literal['cells', 'volume']" = "cells",
    cell_volume: Optional[xr.DataArray] = None,
) -> "Figure":
    """
    One event's occupied cells or volume at each level against time.

    Parameters
    ----------
    events_ds : xarray.Dataset
        4-D events dataset with an ``ID_field`` of dimensions ``(time, zdim, y, x)``.
    event_id : int
        The event to draw.
    zdim : str
        Name of the depth dimension.
    quantity : {"cells", "volume"}
        Occupied cell count, or occupied volume in m^3. ``"volume"`` requires ``cell_volume``.
    cell_volume : xarray.DataArray, optional
        Volume of each grid cell in m^3, with dimensions ``(zdim, y, x)``.

    Returns
    -------
    matplotlib.figure.Figure
        A depth-versus-time section with a colorbar (two axes); depth increases downwards.
    """
    _check_plotting_dependencies()
    if quantity not in ("cells", "volume"):
        raise ConfigurationError(f"quantity must be 'cells' or 'volume', got {quantity!r}")
    if quantity == "volume" and cell_volume is None:
        raise ConfigurationError(
            "quantity='volume' needs the cell volumes",
            details="The events dataset stores per-event totals, not the volume of each grid cell",
            suggestions=["Pass cell_volume, a (depth, lat, lon) array of cell volumes in m^3", "Use quantity='cells'"],
        )
    ids = _id_field(events_ds, zdim)
    ydim, xdim = _horizontal_dims(ids, zdim)
    event_id = _check_event(events_ds, event_id)

    # Restrict to the event's lifetime before reducing; the reduction is lazy and only the
    # (time, level) result is loaded.
    present = events_ds["presence"].sel(ID=event_id).load().values.astype(bool) if "presence" in events_ds else None
    window = ids if present is None or not present.any() else ids.isel({_TIME: np.flatnonzero(present)})
    member = window == event_id
    if quantity == "cells":
        per_level = member.sum([ydim, xdim])
        label = "occupied cells"
    else:
        per_level = (member * cell_volume).sum([ydim, xdim])
        label = "occupied volume (m$^3$)"
    per_level = per_level.transpose(_TIME, zdim).load()

    depths = _depth_values(ids, zdim)
    times, time_label = _time_axis(per_level[_TIME].values)
    data = np.where(per_level.values > 0, per_level.values, np.nan).astype(float)

    fig, ax = plt.subplots(figsize=(7, 4))
    mesh = ax.pcolormesh(times, depths, data.T, shading="nearest", cmap="viridis")
    ax.invert_yaxis()
    ax.set_ylabel(f"{zdim} (m)")
    ax.set_xlabel(time_label)
    ax.set_title(f"Event {event_id}: {label} by level")
    fig.colorbar(mesh, ax=ax, label=label)
    fig.autofmt_xdate()
    return fig


# --------------------------------------------------------------------------------------
# 3. Event summary series
# --------------------------------------------------------------------------------------


def _run_starts(active: np.ndarray) -> np.ndarray:
    """Return the indices where a run of consecutive True values begins."""
    return np.flatnonzero(active & ~np.concatenate([[False], active[:-1]]))


def _merge_steps(events_ds: xr.Dataset, event_id: int) -> np.ndarray:
    """Return the time indices where this event's own merge ledger rows start a merge episode."""
    if "merge_ledger" not in events_ds:
        return np.zeros(0, dtype=int)
    rows = events_ds["merge_ledger"].sel(ID=event_id).load()
    return _run_starts((rows.values >= 0).any(axis=-1))


def _split_steps(time_values: np.ndarray, event_id: int, merges_ds: Optional[xr.Dataset]) -> np.ndarray:
    """Return the time indices where a run of this event's recorded splits begins."""
    if merges_ds is None or "split_event_IDs" not in merges_ds or "split_time" not in merges_ds:
        return np.zeros(0, dtype=int)
    days = merges_ds["split_time"].values[merges_ds["split_event_IDs"].values == event_id]
    return _run_starts(np.isin(time_values, days))


def plot_event_summary(
    events_ds: xr.Dataset,
    event_id: int,
    merges_ds: Optional[xr.Dataset] = None,
) -> "Figure":
    """
    One event's volume, depth centroid and depth extent against time.

    Vertical lines mark where a run of consecutive days with a merge begins, taken from this event's own
    rows of the dataset's ``merge_ledger``, and where a run of splits begins, taken from ``merges_ds``.

    Parameters
    ----------
    events_ds : xarray.Dataset
        4-D events dataset with ``volume``, ``depth_centroid``, ``depth_top``,
        ``depth_bottom`` and ``presence``.
    event_id : int
        The event to draw.
    merges_ds : xarray.Dataset, optional
        The merge dataset from ``run(return_merges=True)``, the source of the split marks. Without
        it, or with one written before split records existed, only merges are marked.

    Returns
    -------
    matplotlib.figure.Figure
        Three stacked axes sharing time: volume, depth centroid, and depth top and bottom.
    """
    _check_plotting_dependencies()
    event_id = _check_event(events_ds, event_id)
    missing = [v for v in ("volume", "depth_centroid", "depth_top", "depth_bottom") if v not in events_ds]
    if missing:
        raise ConfigurationError(
            f"The events dataset lacks {missing}",
            suggestions=["Pass the events dataset returned by the 4-D tracker"],
        )

    def series(name: str) -> xr.DataArray:
        da = events_ds[name].sel(ID=event_id).load()
        if "presence" in events_ds:
            da = da.where(events_ds["presence"].sel(ID=event_id).load())
        return da

    volume, centroid = series("volume"), series("depth_centroid")
    top, bottom = series("depth_top"), series("depth_bottom")
    times, time_label = _time_axis(volume[_TIME].values)

    fig, (ax_v, ax_c, ax_e) = plt.subplots(3, 1, figsize=(7, 7), sharex=True)
    ax_v.plot(times, volume.values, color="tab:blue")
    ax_v.set_ylabel("volume (m$^3$)")
    ax_c.plot(times, centroid.values, color="tab:green")
    ax_c.set_ylabel("depth centroid (m)")
    ax_e.plot(times, top.values, color="tab:orange", label="top")
    ax_e.plot(times, bottom.values, color="tab:red", label="bottom")
    ax_e.fill_between(times, top.values, bottom.values, color="tab:orange", alpha=0.15)
    ax_e.set_ylabel("depth extent (m)")
    ax_e.legend(loc="best", fontsize=8)
    for ax in (ax_c, ax_e):
        ax.invert_yaxis()  # depth is positive down

    for i, step in enumerate(_merge_steps(events_ds, event_id)):
        for ax in (ax_v, ax_c, ax_e):
            ax.axvline(
                times[step], color="tab:purple", linestyle="--", linewidth=1.0, label="merge" if i == 0 and ax is ax_v else None
            )
    for i, step in enumerate(_split_steps(volume[_TIME].values, event_id, merges_ds)):
        for ax in (ax_v, ax_c, ax_e):
            ax.axvline(
                times[step], color="tab:brown", linestyle=":", linewidth=1.2, label="split" if i == 0 and ax is ax_v else None
            )
    if ax_v.get_legend_handles_labels()[1]:
        ax_v.legend(loc="best", fontsize=8)

    ax_v.set_title(f"Event {event_id}")
    ax_e.set_xlabel(time_label)
    fig.autofmt_xdate()
    return fig


# --------------------------------------------------------------------------------------
# 4. Animation
# --------------------------------------------------------------------------------------


def _check_levels(ids: xr.DataArray, zdim: str, levels: Any) -> List[int]:
    """Validate a sequence of level indices against the depth dimension."""
    picks = [int(k) for k in levels]
    if not picks:
        raise ConfigurationError("levels is empty", suggestions=["Pass level indices, or levels='footprint'"])
    nz = ids.sizes[zdim]
    bad = [k for k in picks if not -nz <= k < nz]
    if bad:
        raise ConfigurationError(f"Level indices {bad} are outside the {nz} levels of '{zdim}'")
    return picks


def animate_event_levels(
    events_ds: xr.Dataset,
    *,
    zdim: str,
    levels: "Union[Sequence[int], Literal['footprint']]" = "footprint",
    plot_dir: str = "./",
    file_name: Optional[str] = None,
    config: Optional[PlotConfig] = None,
) -> Optional[str]:
    """
    Animate the events through time, as a depth-projected footprint or as one map panel per level.

    ``levels="footprint"`` shows where any event occupies the water column, coloured by the depth of
    the shallowest occupied level. A sequence of level indices instead draws one map panel per
    level, each coloured by event ID. In both modes the canvas size is derived once from the
    domain, so every frame has the same pixel dimensions (a requirement of the h264 encoder).

    Parameters
    ----------
    events_ds : xarray.Dataset
        4-D events dataset with an ``ID_field`` of dimensions ``(time, zdim, y, x)``.
    zdim : str
        Name of the depth dimension.
    levels : sequence of int or "footprint"
        ``"footprint"`` projects over every level. A sequence of level indices gives one panel
        per level.
    plot_dir : str
        Directory the movie is written to.
    file_name : str, optional
        Movie file name; ``.mp4`` is appended when missing.
    config : PlotConfig, optional
        Plot configuration (projection, framerate, colormap, colorbar).

    Returns
    -------
    str or None
        Path of the written movie, or ``None`` when ``ffmpeg`` is not installed.
    """
    _check_plotting_dependencies()
    config = config or PlotConfig()
    ids = _id_field(events_ds, zdim)
    ydim, xdim = _horizontal_dims(ids, zdim)

    if not isinstance(levels, str):
        picks = _check_levels(ids, zdim, levels)
        return _animate_level_panels(ids, picks, zdim, ydim, xdim, config, plot_dir, file_name)
    if levels != "footprint":
        raise ConfigurationError(f"levels must be 'footprint' or a sequence of level indices, got {levels!r}")

    # Shallowest occupied depth per column, lazily: depth where occupied, +inf elsewhere, min over
    # depth, then +inf (no event in the column) back to missing.
    depth_values = _depth_values(ids, zdim)
    depth = xr.DataArray(depth_values, dims=[zdim])
    shallowest = xr.where(ids > 0, depth, np.inf).min(zdim)
    shallowest = shallowest.where(np.isfinite(shallowest)).transpose(_TIME, ydim, xdim)
    shallowest.name = "event_footprint_depth"

    lo, hi = float(depth_values.min()), float(depth_values.max())
    anim_config = PlotConfig(
        title=config.title,
        var_units=config.var_units or f"shallowest occupied {zdim} (m)",
        cmap=config.cmap or "viridis_r",
        clim=config.clim or (lo, hi if hi > lo else lo + 1.0),
        show_colorbar=config.show_colorbar,
        grid_lines=config.grid_lines,
        grid_labels=config.grid_labels,
        dimensions={"time": _TIME, "y": ydim, "x": xdim},
        coordinates={"time": _TIME, "y": ydim, "x": xdim},
        projection=config.projection,
        framerate=config.framerate,
        extend="neither",
    )
    plotter = GriddedPlotter(shallowest, anim_config.dimensions, anim_config.coordinates)
    return _animate(plotter, anim_config, plot_dir=plot_dir, file_name=file_name or "event_footprint_depth")


def _panel_canvas(projection: Any, lons: np.ndarray, lats: np.ndarray, n: int) -> Tuple[float, float, int, int]:
    """Return ``(width, height, nrows, ncols)`` of a canvas of ``n`` panels, in inches, sized from the domain."""
    ncols = min(_PANEL_COLUMNS, n)
    nrows = int(np.ceil(n / ncols))
    # One panel's aspect, as the projection draws the domain; the canvas is that grid of panels.
    pw, ph = _anim._domain_figsize(projection, lons, lats, False)
    width = 7.0 if ncols == 1 else 12.0
    panel_w = width / ncols
    height = nrows * (panel_w * ph / pw + 0.5) + 0.6
    # Keep both sides under the h264 pixel ceiling and on even pixel counts.
    limit = _anim._MAX_FRAME_PIXELS / _anim._FRAME_DPI
    scale = min(1.0, limit / width, limit / height)
    px_w = int(round(width * scale * _anim._FRAME_DPI)) // 2 * 2
    px_h = int(round(height * scale * _anim._FRAME_DPI)) // 2 * 2
    return px_w / _anim._FRAME_DPI, px_h / _anim._FRAME_DPI, nrows, ncols


def _animate_level_panels(
    ids: xr.DataArray,
    picks: List[int],
    zdim: str,
    ydim: str,
    xdim: str,
    config: PlotConfig,
    plot_dir: Any,
    file_name: Optional[str],
) -> Optional[str]:
    """Write a movie with one map panel per level; every frame is drawn on the same canvas."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    name = file_name or "event_levels"
    projection = config.projection or ccrs.Robinson()
    lons, lats = ids[xdim].values, ids[ydim].values
    depths = _depth_values(ids, zdim)[picks]
    times = ids[_TIME].values
    cmap, norm = _id_style(None)

    # Sized once, here, and reused for every frame.
    width, height, nrows, ncols = _panel_canvas(projection, lons, lats, len(picks))

    def render_frames(temp_dir: Path) -> None:
        for t in range(ids.sizes[_TIME]):
            # Select the time step and levels first; only this block is loaded.
            block = ids.isel({_TIME: t, zdim: picks}).transpose(zdim, ydim, xdim).values
            fig = Figure(figsize=(width, height))
            FigureCanvasAgg(fig)
            try:
                axes = fig.subplots(nrows, ncols, squeeze=False, subplot_kw={"projection": projection}).flatten()
                for i, ax in enumerate(axes):
                    if i >= len(picks):
                        fig.delaxes(ax)
                        continue
                    pcolormesh_seam_safe(
                        ax,
                        lons,
                        lats,
                        _id_values(block[i], None),
                        transform=ccrs.PlateCarree(),
                        cmap=cmap,
                        norm=norm,
                        shading="auto",
                    )
                    _add_map_features(ax, config.grid_lines)
                    ax.set_title(f"{zdim} = {depths[i]:.4g} m", size=9)
                fig.suptitle(_time_text(times[t]), size=12)
                # No bbox_inches="tight": it crops to content and the frame size would drift.
                fig.savefig(temp_dir / f"time_{t:04d}.jpg", dpi=_anim._FRAME_DPI, pil_kwargs={"quality": 95})
            finally:
                fig.clear()
            gc.collect()

    return _anim._write_movie(plot_dir, name, config.framerate, render_frames)
