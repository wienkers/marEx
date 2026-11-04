"""
Plotting helpers for 4-D ``(time, depth, lat, lon)`` event output.

Four helpers draw the events dataset a 4-D tracker returns:

* :func:`plot_event_levels`: map panels of the event IDs at selected depth levels for one time step
* :func:`plot_event_depth_time`: one event's occupied cell count or volume per level against time
* :func:`plot_event_summary`: one event's volume, depth centroid and depth extent against time, with
  merges and splits marked
* :func:`animate_event_levels`: the depth-projected footprint coloured by shallowest occupied depth

``plot_event_levels`` and ``animate_event_levels`` take an optional ``extent`` to zoom the maps onto the
events, and ``plot_event_levels`` can zoom onto the highlighted event by itself.

Every helper selects the time steps, levels or event it needs from the lazy ID field before loading
anything, so a field larger than memory is never materialised to draw one panel.
"""

import gc
import warnings
from pathlib import Path
from typing import Any, List, Optional, Sequence, Tuple, Union

import numpy as np
import xarray as xr

from ..exceptions import ConfigurationError
from ..logging_config import get_logger
from . import animation as _anim
from .config import PlotConfig
from .dependencies import _check_plotting_dependencies
from .mesh import pcolormesh_seam_safe

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import matplotlib.pyplot as plt
    from matplotlib.colors import ListedColormap, Normalize
    from matplotlib.figure import Figure
    from matplotlib.ticker import FuncFormatter
except ImportError:  # pragma: no cover - checked by _check_plotting_dependencies at call time
    ccrs = None
    cfeature = None
    plt = None
    ListedColormap = None
    Normalize = None
    FuncFormatter = None
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
# How a drawn map window compares with the same longitude range centred on the projection: see _warn_if_extent_misdrawn.
_COLLAPSED_FRACTION = 0.5
_WIDER_THAN_FRAMED = 1.02
_FULL_WIDTH_FRACTION = 0.7
_WHOLE_WIDTH_FRACTION = 0.9999
_NEARLY_A_WHOLE_TURN = 359.0  # the full width is what such a range asks for


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
    return ListedColormap(["#8a94a0", "tab:red"]), Normalize(-0.5, 1.5)


def _id_values(level_ids: np.ndarray, event_id: Optional[int]) -> np.ndarray:
    """Return the colour index drawn for each cell of one level (NaN where no event)."""
    if event_id is None:
        return np.where(level_ids > 0, (level_ids - 1) % _N_ID_COLOURS, np.nan).astype(float)
    return np.where(level_ids == event_id, 1.0, np.where(level_ids > 0, 0.0, np.nan))


# Map colours: a light ocean and darker land, so events in any colour map stand out and nothing is left white.
# The "dark" theme is a slate ocean with darker land, for figures on a dark background.
_THEMES = {
    "light": {"background": "white", "text": "black", "ocean": "#dde6ee", "land": "#9aa3ad", "coast": "black", "grid": "white"},
    "dark": {
        "background": "#111925",
        "text": "#dbe4ee",
        "ocean": "#2b3a4d",
        "land": "#0d131c",
        "coast": "#8da0b5",
        "grid": "#9fb0c3",
    },
}


def _check_theme(theme: str) -> str:
    if theme not in _THEMES:
        raise ConfigurationError(f"theme must be 'light' or 'dark', got {theme!r}")
    return theme


def _add_map_features(
    ax: Any, grid_lines: bool, extent: Optional[Sequence[float]] = None, theme: str = "light", grid_labels: bool = False
) -> None:
    """Ocean and land colours, coastlines, an optional light graticule (labelled if asked) and an optional zoom."""
    colours = _THEMES[theme]
    ax.set_facecolor(colours["ocean"])
    ax.add_feature(cfeature.LAND.with_scale("50m"), facecolor=colours["land"], zorder=2)
    ax.add_feature(cfeature.COASTLINE.with_scale("50m"), linewidth=0.4, edgecolor=colours["coast"], zorder=3)
    if grid_lines:
        graticule = ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=grid_labels,
            linewidth=0.4,
            color=colours["grid"],
            alpha=0.35,
            linestyle="-",
            zorder=4,
        )
        graticule.xlabel_style = {"color": colours["text"]}
        graticule.ylabel_style = {"color": colours["text"]}
    if extent is not None:
        ax.set_extent(list(extent), crs=ccrs.PlateCarree())


def _check_extent(extent: Optional[Sequence[float]]) -> Optional[Tuple[float, float, float, float]]:
    """Validate a ``(lon_min, lon_max, lat_min, lat_max)`` extent in degrees, or pass ``None`` through."""
    if extent is None:
        return None
    try:
        lon0, lon1, lat0, lat1 = (float(v) for v in extent)
    except (TypeError, ValueError):
        raise ConfigurationError(
            f"extent must be four numbers (lon_min, lon_max, lat_min, lat_max), got {extent!r}",
            suggestions=["Pass extent=(lon_min, lon_max, lat_min, lat_max) in degrees"],
        ) from None
    if not (lon0 < lon1 and lat0 < lat1):
        raise ConfigurationError(f"extent must have lon_min < lon_max and lat_min < lat_max, got {extent!r}")
    return lon0, lon1, lat0, lat1


def _event_extent(member: np.ndarray, lons: np.ndarray, lats: np.ndarray) -> Optional[Tuple[float, float, float, float]]:
    """
    Bounding box of a non-empty ``(y, x)`` boolean mask, padded by 15% of its span and at least 2 degrees.

    On a grid that closes around the globe the box stays on the grid, because a longitude range past the seam wraps
    and can leave the event out of view. An event in both edge columns of such a grid, or one whose box fills the
    width, has no window short of the whole domain: this warns and returns ``None``.
    """
    rows, cols = np.flatnonzero(member.any(axis=1)), np.flatnonzero(member.any(axis=0))
    lon0, lon1 = sorted((lons[cols[0]], lons[cols[-1]]))
    lat0, lat1 = sorted((lats[rows[0]], lats[rows[-1]]))
    pad = max(2.0, 0.15 * max(lon1 - lon0, lat1 - lat0))
    west, east = lon0 - pad, lon1 + pad
    low, high = lons.min(), lons.max()
    closes = lons.size > 1 and high - low + abs(lons[1] - lons[0]) >= 359.99
    if closes or east - west >= 360.0:
        west, east = max(west, low), min(east, high)
    if (closes and cols[0] == 0 and cols[-1] == lons.size - 1) or east - west >= 359.0:
        logger.warning(
            "The event fills the grid's width or lies on both sides of its longitude seam, so zoom_to_event keeps "
            "the whole domain in view. Pass extent and a projection with central_longitude at the event's centre to "
            "frame it"
        )
        return None
    return float(west), float(east), float(max(lat0 - pad, -90.0)), float(min(lat1 + pad, 90.0))


def _drawn_width(extent: Tuple[float, float, float, float], projection: Any) -> float:
    """Width, in projection units, of the window ``set_extent`` draws for this extent, on a throwaway axes."""
    ax = Figure().add_subplot(projection=projection)
    with warnings.catch_warnings():
        warnings.simplefilter("ignore")  # a collapsed window warns on the real axes too
        ax.set_extent(list(extent), crs=ccrs.PlateCarree())
    x0, x1 = ax.get_xlim()
    return abs(x1 - x0)


def _has_longitude_edge(projection: Any, centre: float) -> bool:
    """
    Whether the meridian opposite the map's centre splits it at the equator: its two sides land at opposite edges.

    Equator-centred azimuthal and rotated maps count as split, though a window away from the equator can still frame.
    """
    plate = ccrs.PlateCarree()
    west = projection.transform_point(centre - 179.9, 0.0, plate)[0]
    east = projection.transform_point(centre + 179.9, 0.0, plate)[0]
    return bool(abs(east - west) > 0.9 * abs(projection.x_limits[1] - projection.x_limits[0]))


def _warn_if_extent_misdrawn(extent: Optional[Tuple[float, float, float, float]], projection: Any) -> None:
    """
    Warn when the projection may not frame this longitude range, from what the map draws for it.

    A range wider than a whole turn is cropped on some maps. A range of a whole turn can collapse to almost nothing. A
    range that crosses the meridian opposite the centre, on a map that this meridian splits at the equator, is drawn in
    the wrong place even when the width looks right. A range that ends on that meridian may frame or not, so it is judged
    by the drawn width: the same range centred on the projection holds no seam, and a window wider than that, near the
    map's full width at these latitudes, has spread across it. The
    full-width test keeps a narrow window near the map's edge, which a curved projection draws somewhat wider, out.
    Some windows on azimuthal and rotated maps warn although they frame, see :func:`_has_longitude_edge`.
    """
    if extent is None:
        return
    lon0, lon1, lat0, lat1 = extent
    asked = lon1 - lon0
    centre = float(projection.proj4_params.get("lon_0", 0.0))
    half = min(asked, 360.0) / 2.0
    try:
        drawn = _drawn_width(extent, projection)
        framed = _drawn_width((centre - half, centre + half, lat0, lat1), projection)
        widest = _drawn_width((centre - 180.0, centre + 180.0, lat0, lat1), projection)
        has_edge = _has_longitude_edge(projection, centre)
    except Exception:  # only a check: the real set_extent raises its own error if the extent is unusable
        return
    next_seam = centre + 180.0 + 360.0 * (np.floor((lon0 - centre - 180.0) / 360.0) + 1.0)
    crosses = next_seam < lon1 and asked < _NEARLY_A_WHOLE_TURN and has_edge
    midpoint = (lon0 + lon1) / 2.0
    advice = f"Try a projection with central_longitude={180.0 - (180.0 - midpoint) % 360.0:g}"
    if asked > 360.0:
        if drawn < _WHOLE_WIDTH_FRACTION * widest:
            logger.warning(f"extent {extent[:2]} is wider than a whole turn, so the map crops it")
    elif asked >= 360.0 and drawn < _COLLAPSED_FRACTION * framed:
        logger.warning(f"extent {extent[:2]} wraps around the projection's longitude seam, so the map collapses. {advice}")
    elif crosses or (drawn > _WIDER_THAN_FRAMED * framed and drawn >= _FULL_WIDTH_FRACTION * widest):
        logger.warning(
            f"extent {extent[:2]} crosses or ends on the projection's longitude seam, so the map may draw the wrong window. "
            f"{advice}"
        )


def _depth_limits(depths: np.ndarray) -> Tuple[float, float]:
    """
    Depth axis limits, deep end first, that hold the whole cell of every level.

    The axis starts at the surface, unless the levels are a crop whose top cell starts more than a level spacing below it.
    """
    if depths.size < 2:
        return float(depths.max()), 0.0
    # By distance from the surface, so levels listed deep-first and depths counted positive up both work.
    levels = depths[np.argsort(np.abs(depths))]
    deep = levels[-1] + (levels[-1] - levels[-2]) / 2.0
    spacing = levels[1] - levels[0]
    top = levels[0] - spacing / 2.0
    return float(deep), float(top) if abs(top) > abs(spacing) else 0.0


def _depth_text(depth: float) -> str:
    """Round a depth in metres for a title: whole metres from 10 m, one decimal in the surface layers."""
    return f"{depth:.0f}" if abs(depth) >= 10 else f"{depth:.1f}"


def _format_time_axis(fig: Any, ax: Any, times: np.ndarray) -> None:
    """Compact date labels on a datetime axis; other calendars keep their elapsed-days axis as it is."""
    if times.dtype.kind != "M":
        return
    import matplotlib.dates as mdates

    locator = mdates.AutoDateLocator()
    ax.xaxis.set_major_locator(locator)
    ax.xaxis.set_major_formatter(mdates.ConciseDateFormatter(locator))


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
    extent: Optional[Sequence[float]] = None,
    zoom_to_event: bool = False,
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
    extent : sequence of 4 floats, optional
        Zoom every panel to ``(lon_min, lon_max, lat_min, lat_max)`` in degrees. The default shows the whole domain.
        A range that crosses or wraps around the longitude seam can draw the wrong window. A warning then suggests a
        ``central_longitude``, for a range of up to 360 degrees.
    zoom_to_event : bool
        Zoom to the highlighted event, with a margin, wherever it occupies the levels drawn at this time step. Needs
        ``event_id``, and ``extent`` takes precedence. An event absent at that time and those levels leaves the whole
        domain in view, with a warning. So does an event on both sides of the grid's longitude seam, or one that fills
        its width, so pass ``extent`` to frame it. The default is False.

    Returns
    -------
    matplotlib.figure.Figure
        One map axes per level. Titles give each depth in metres, rounded.
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
    if zoom_to_event and event_id is None:
        raise ConfigurationError("zoom_to_event needs an event_id", suggestions=["Pass event_id, or pass extent"])
    extent = _check_extent(extent)

    # Select the time step and levels first; only this small block is loaded.
    block = _select_time(ids, time).isel({zdim: levels}).transpose(zdim, ydim, xdim).load()
    depths = _depth_values(ids, zdim)[levels]
    if extent is None and zoom_to_event:
        member = (block.values == event_id).any(axis=0)
        if member.any():
            extent = _event_extent(member, block[xdim].values, block[ydim].values)
        else:
            logger.warning(f"Event {event_id} has no cells at this time and these levels, so the whole domain stays in view")
    time_label = _time_text(block[_TIME].values[()]) if _TIME in block.coords else ""

    proj = projection if projection is not None else config.projection
    proj = proj if proj is not None else ccrs.Robinson()
    _warn_if_extent_misdrawn(extent, proj)
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
        _add_map_features(ax, config.grid_lines, extent)
        ax.set_title(f"{zdim} = {_depth_text(depths[i])} m", size=10)

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
    cmap: str = "magma",
    depth_scale: "Literal['linear', 'sqrt']" = "linear",
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
    cmap : str
        Colour map name. The default is ``"magma"``.
    depth_scale : {"linear", "sqrt"}
        ``"sqrt"`` gives the thin upper levels more room on the depth axis. The default is ``"linear"``.
        Either way the depth axis holds every level of the dataset, whatever depths the event occupies, from the
        surface (0 m), or on a crop from the top of its shallowest level, to the bottom of the deepest.

    Returns
    -------
    matplotlib.figure.Figure
        A depth-versus-time section with a colorbar (two axes); depth increases downwards. The colour bar
        counts in units of 10^n m^3 (or cells), with the factor in its label, so no offset text is drawn.
    """
    _check_plotting_dependencies()
    if quantity not in ("cells", "volume"):
        raise ConfigurationError(f"quantity must be 'cells' or 'volume', got {quantity!r}")
    if depth_scale not in ("linear", "sqrt"):
        raise ConfigurationError(f"depth_scale must be 'linear' or 'sqrt', got {depth_scale!r}")
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
    # Count in units of 10^n, so the colour bar needs no "1e13" offset text. The mesh keeps the raw values.
    exponent = int(np.floor(np.log10(np.nanmax(data)))) if np.isfinite(data).any() else 0
    exponent = exponent if exponent >= 4 else 0  # small counts read as they are
    if exponent:
        label = label.replace("occupied volume (m$^3$)", f"occupied volume (10$^{{{exponent}}}$ m$^3$)").replace(
            "occupied cells", f"occupied cells (10$^{{{exponent}}}$)"
        )

    fig, ax = plt.subplots(figsize=(7, 4), layout="constrained")
    mesh = ax.pcolormesh(times, depths, data.T, shading="nearest", cmap=cmap)
    limits = _depth_limits(depths)
    ax.set_ylim(*limits)
    if depth_scale == "sqrt":
        ax.set_yscale("function", functions=(lambda d: np.sqrt(np.clip(d, 0.0, None)), np.square))
        ax.set_ylim(*limits)
    ax.set_ylabel(f"{zdim} (m)")
    ax.set_xlabel(time_label)
    ax.set_title(f"Event {event_id}: {label} by level")
    colourbar = fig.colorbar(mesh, ax=ax, label=label)
    if exponent:
        colourbar.formatter = FuncFormatter(lambda value, _: f"{value / 10.0**exponent:g}")
        colourbar.update_ticks()
    _format_time_axis(fig, ax, per_level[_TIME].values)
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

    fig, (ax_v, ax_c, ax_e) = plt.subplots(3, 1, figsize=(7, 7), sharex=True, layout="constrained")
    ax_v.plot(times, volume.values, color="tab:blue")
    peak = float(np.nanmax(volume.values)) if np.isfinite(volume.values).any() else 0.0
    exponent = int(np.floor(np.log10(peak))) if peak > 0 else 0
    if exponent >= 4:  # in units of 10^n, so the axis needs no "1e14" offset text
        ax_v.yaxis.set_major_formatter(FuncFormatter(lambda value, _: f"{value / 10.0**exponent:g}"))
        ax_v.set_ylabel(f"volume (10$^{{{exponent}}}$ m$^3$)")
    else:
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
    _format_time_axis(fig, ax_e, volume[_TIME].values)
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
    extent: Optional[Sequence[float]] = None,
    theme: "Literal['light', 'dark']" = "light",
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
        Plot configuration (projection, framerate, colormap, colorbar, grid lines).
    extent : sequence of 4 floats, optional
        Zoom to ``(lon_min, lon_max, lat_min, lat_max)`` in degrees, with the canvas sized to that window, so the
        events fill the frame. The default shows the whole domain. A range the map cannot frame, because it crosses or
        wraps around the longitude seam, can draw the wrong window, and a warning suggests a ``central_longitude``,
        for a range of up to 360 degrees. ``PlotConfig`` supplies ``Robinson`` unless a projection is set.
    theme : {"light", "dark"}
        ``"dark"`` draws a slate ocean, dark land and light text on a dark background, for use beside dark figures.
        The default is ``"light"``.

    Returns
    -------
    str or None
        Path of the written movie, or ``None`` when ``ffmpeg`` is not installed.
    """
    _check_plotting_dependencies()
    config = config or PlotConfig()
    ids = _id_field(events_ds, zdim)
    ydim, xdim = _horizontal_dims(ids, zdim)
    extent = _check_extent(extent)
    theme = _check_theme(theme)
    _warn_if_extent_misdrawn(extent, config.projection or ccrs.Robinson())

    if not isinstance(levels, str):
        picks = _check_levels(ids, zdim, levels)
        return _animate_level_panels(ids, picks, zdim, ydim, xdim, config, plot_dir, file_name, extent, theme)
    if levels != "footprint":
        raise ConfigurationError(f"levels must be 'footprint' or a sequence of level indices, got {levels!r}")

    # Shallowest occupied depth per column, lazily: depth where occupied, +inf elsewhere, min over
    # depth, then +inf (no event in the column) back to missing.
    depth_values = _depth_values(ids, zdim)
    shallowest = _shallowest_occupied(ids, zdim, ydim, xdim)
    lo, hi = float(depth_values.min()), float(depth_values.max())
    clim = config.clim or (lo, hi if hi > lo else lo + 1.0)
    return _animate_footprint(
        shallowest, zdim, ydim, xdim, config, clim, plot_dir, file_name or "event_footprint_depth", extent, theme
    )


def _shallowest_occupied(ids: xr.DataArray, zdim: str, ydim: str, xdim: str) -> xr.DataArray:
    """Depth of the shallowest occupied level of each column at each time step (NaN where no event), still lazy."""
    depth = xr.DataArray(_depth_values(ids, zdim), dims=[zdim])
    shallowest = xr.where(ids > 0, depth, np.inf).min(zdim)
    shallowest = shallowest.where(np.isfinite(shallowest)).transpose(_TIME, ydim, xdim)
    shallowest.name = "event_footprint_depth"
    return shallowest


def _animate_footprint(
    shallowest: xr.DataArray,
    zdim: str,
    ydim: str,
    xdim: str,
    config: PlotConfig,
    clim: Tuple[float, float],
    plot_dir: Any,
    name: str,
    extent: Optional[Tuple[float, float, float, float]],
    theme: str = "light",
) -> Optional[str]:
    """Write the footprint movie: events over a land and ocean map, one canvas size for every frame."""
    from matplotlib.backends.backend_agg import FigureCanvasAgg

    projection = config.projection or (ccrs.PlateCarree() if extent else ccrs.Robinson())
    lons, lats = shallowest[xdim].values, shallowest[ydim].values
    window_lons, window_lats = (np.array(extent[:2]), np.array(extent[2:])) if extent else (lons, lats)
    cmap = plt.get_cmap(config.cmap or "viridis_r")
    norm = Normalize(*clim)
    times = shallowest[_TIME].values
    # Sized once, here, and reused for every frame. Room for the title above and the colour bar below.
    width, height, _, _ = _panel_canvas(projection, window_lons, window_lats, 1, extra_height=0.8 if config.show_colorbar else 0.0)

    def render_frames(temp_dir: Path) -> None:
        for t in range(shallowest.sizes[_TIME]):
            # One time step at a time: only this (y, x) slice is computed.
            values = shallowest.isel({_TIME: t}).values
            fig = Figure(figsize=(width, height), facecolor=_THEMES[theme]["background"])
            FigureCanvasAgg(fig)
            try:
                ax = fig.add_axes([0.03, 0.2 if config.show_colorbar else 0.04, 0.94, 0.72], projection=projection)
                pcolormesh_seam_safe(ax, lons, lats, values, transform=ccrs.PlateCarree(), cmap=cmap, norm=norm, shading="auto")
                _add_map_features(ax, config.grid_lines, extent, theme, grid_labels=config.grid_labels)
                if config.show_colorbar:
                    cax = fig.add_axes([0.2, 0.08, 0.6, 0.035])
                    bar = fig.colorbar(plt.cm.ScalarMappable(norm=norm, cmap=cmap), cax=cax, orientation="horizontal")
                    bar.set_label(config.var_units or f"shallowest occupied {zdim} (m)", size=9, color=_THEMES[theme]["text"])
                    bar.ax.tick_params(colors=_THEMES[theme]["text"])
                title = _time_text(times[t])
                fig.suptitle(f"{config.title}, {title}" if config.title else title, size=12, color=_THEMES[theme]["text"])
                # No bbox_inches="tight": it crops to content and the frame size would drift.
                fig.savefig(temp_dir / f"time_{t:04d}.jpg", dpi=_anim._FRAME_DPI, pil_kwargs={"quality": 95})
            finally:
                fig.clear()
            gc.collect()

    return _anim._write_movie(plot_dir, name, config.framerate, render_frames)


def _panel_canvas(
    projection: Any, lons: np.ndarray, lats: np.ndarray, n: int, extra_height: float = 0.0
) -> Tuple[float, float, int, int]:
    """Return ``(width, height, nrows, ncols)`` of a canvas of ``n`` panels, in inches, sized from the domain."""
    ncols = min(_PANEL_COLUMNS, n)
    nrows = int(np.ceil(n / ncols))
    # One panel's aspect, as the projection draws the domain; the canvas is that grid of panels.
    pw, ph = _anim._domain_figsize(projection, lons, lats, False)
    width = 7.0 if ncols == 1 else 12.0
    panel_w = width / ncols
    height = nrows * (panel_w * ph / pw + 0.5) + 0.6 + extra_height
    # Keep both sides under the h264 pixel ceiling and on even pixel counts.
    limit = _anim._MAX_FRAME_PIXELS / _anim._FRAME_DPI
    scale = min(1.0, limit / width, limit / height)
    px_w = int(round(width * scale * _anim._FRAME_DPI)) // 2 * 2
    px_h = int(round(height * scale * _anim._FRAME_DPI)) // 2 * 2
    # A hundredth of a pixel over, so the float product with the dpi never truncates an even pixel count to an odd one.
    return (px_w + 0.01) / _anim._FRAME_DPI, (px_h + 0.01) / _anim._FRAME_DPI, nrows, ncols


def _animate_level_panels(
    ids: xr.DataArray,
    picks: List[int],
    zdim: str,
    ydim: str,
    xdim: str,
    config: PlotConfig,
    plot_dir: Any,
    file_name: Optional[str],
    extent: Optional[Tuple[float, float, float, float]] = None,
    theme: str = "light",
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
    window_lons, window_lats = (np.array(extent[:2]), np.array(extent[2:])) if extent else (lons, lats)
    width, height, nrows, ncols = _panel_canvas(projection, window_lons, window_lats, len(picks))

    def render_frames(temp_dir: Path) -> None:
        for t in range(ids.sizes[_TIME]):
            # Select the time step and levels first; only this block is loaded.
            block = ids.isel({_TIME: t, zdim: picks}).transpose(zdim, ydim, xdim).values
            fig = Figure(figsize=(width, height), facecolor=_THEMES[theme]["background"])
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
                    _add_map_features(ax, config.grid_lines, extent, theme)
                    ax.set_title(f"{zdim} = {_depth_text(depths[i])} m", size=9, color=_THEMES[theme]["text"])
                fig.suptitle(_time_text(times[t]), size=12, color=_THEMES[theme]["text"])
                # No bbox_inches="tight": it crops to content and the frame size would drift.
                fig.savefig(temp_dir / f"time_{t:04d}.jpg", dpi=_anim._FRAME_DPI, pil_kwargs={"quality": 95})
            finally:
                fig.clear()
            gc.collect()

    return _anim._write_movie(plot_dir, name, config.framerate, render_frames)
