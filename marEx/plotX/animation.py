"""Animation utilities for the plotX visualisation system.

Provides the ``make_frame`` dask-delayed frame renderer and the ``_animate``
orchestrator that drives it. ``PlotterBase.animate`` is a thin delegator to
``_animate`` in this module.
"""

import gc
import shutil
import subprocess
import tempfile
import warnings
from pathlib import Path
from typing import Any, Dict, Optional, Union

import dask
import numpy as np
import xarray as xr

from ..exceptions import DependencyError, VisualisationError
from ..logging_config import get_logger

# Handle optional dependencies for plotting
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    ccrs = None
    cfeature = None

# Use the object-oriented Figure / FigureCanvasAgg API rather than the pyplot global
# state machine: frame rendering runs on Dask's (default) threaded scheduler, and the
# pyplot state machine is not thread-safe.
try:
    from matplotlib.backends.backend_agg import FigureCanvasAgg
    from matplotlib.figure import Figure

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    Figure = None
    FigureCanvasAgg = None

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None

logger = get_logger(__name__)


# Frame geometry.
#
# Every frame in a movie must have identical pixel dimensions -- h264 rejects a stream whose
# frame size changes mid-way -- which is why ``bbox_inches="tight"`` is not used when saving
# frames: it crops to content, so the dimensions drift with the data.
#
# A single hardcoded ``figsize`` satisfies that constraint but wastes most of the frame: a
# global map is roughly 2:1, so drawing it on a fixed 7x5 canvas left ~45% of the frame as
# white margin. The fix is not to crop per frame (that reintroduces the drift) but to choose
# ONE figure aspect up front, from the projected extent of the domain, and use it for every
# frame. The domain does not change between frames, so the dimensions stay constant.
_FRAME_DPI = 300
_FIG_WIDTH_INCHES = 7.0
# Fraction of the figure the axes box occupies, from matplotlib's default subplot margins
# (left=0.125, right=0.9, bottom=0.11, top=0.88). The top margin also holds the date title.
_AXES_WIDTH_FRACTION = 0.775
_AXES_HEIGHT_FRACTION = 0.77
# A ``shrink=0.6`` colorbar plus its padding takes roughly a fifth of the width from the map.
_COLORBAR_WIDTH_FRACTION = 0.80
# Lower guard rail against a domain so wide it stops being usable -- a narrow zonal band would
# otherwise ask for a frame a few dozen pixels tall. Keep it low: a legitimate 4:1 domain (say
# 200 deg by 50 deg) lands near 0.25, so a floor of 0.30 would clamp ordinary data and quietly
# reintroduce the whitespace this sizing exists to remove.
_MIN_FIG_ASPECT = 0.15
# Upper guard rail is a PIXEL ceiling, not an aspect one. H.264 levels cap a single dimension
# at 4096 px, and browsers and most hardware decoders enforce it -- but libx264 encodes larger
# without complaint, so an oversized frame fails at playback rather than at write time, in the
# browser, which is exactly where these files end up. A tall narrow domain reaches it easily:
# 10 deg lon by 80 deg lat projects to aspect ~8. Expressed as an aspect this was 2.00, which
# yields 7 x 14 in at 300 dpi = 2100x4200 -- just over the line.
_MAX_FRAME_PIXELS = 4096
_DEFAULT_FIG_ASPECT = 5.0 / 7.0


def _even_height(height_inches: float) -> float:
    """Round a figure height so ``height * _FRAME_DPI`` lands on an even pixel count.

    h264 needs even dimensions. ``make_frame`` already crops odd frames as a safety net, but
    landing on even avoids re-opening and rewriting every frame.
    """
    pixels = max(2, int(round(height_inches * _FRAME_DPI)))
    if pixels % 2:
        pixels += 1
    return pixels / _FRAME_DPI


def _domain_figsize(projection, x_values, y_values, show_colorbar: bool):
    """Pick a figure size whose aspect matches the domain *as the projection will draw it*.

    The aspect has to come from projected coordinates, not raw lon/lat: Robinson draws a
    global domain at about 2:1, while the underlying lon/lat rectangle is 360x180 = 2:1 only
    by coincidence, and any other projection differs. Returns the module default unchanged if
    the extent cannot be established, so an unusual grid degrades to previous behaviour
    rather than raising from inside a render.
    """
    default = (_FIG_WIDTH_INCHES, _even_height(_FIG_WIDTH_INCHES * _DEFAULT_FIG_ASPECT))

    def _fallback(reason: str):
        # Silent degradation is this package's most expensive recurring failure mode: the
        # frames would simply come back with the old whitespace and nothing would say why.
        logger.debug(f"Animation frame sizing fell back to the default canvas ({reason}); frames may show extra margin")
        return default

    if x_values is None or y_values is None:
        return _fallback("x/y coordinates not available on the DataArray")

    try:
        x = np.asarray(x_values, dtype=float).ravel()
        y = np.asarray(y_values, dtype=float).ravel()
        if x.size == 0 or y.size == 0:
            return _fallback("empty coordinate array")

        # Sample the interior, not just the four corners: for a curved projection the
        # projected bounding box of a lon/lat rectangle is not attained at its corners
        # (Robinson bulges at the equator and tapers towards the poles).
        lon_samples = np.linspace(float(np.nanmin(x)), float(np.nanmax(x)), 25)
        lat_samples = np.linspace(float(np.nanmin(y)), float(np.nanmax(y)), 25)
        grid_lon, grid_lat = np.meshgrid(lon_samples, lat_samples)

        projected = projection.transform_points(ccrs.PlateCarree(), grid_lon.ravel(), grid_lat.ravel())
        px, py = projected[:, 0], projected[:, 1]
        finite = np.isfinite(px) & np.isfinite(py)
        if finite.sum() < 2:
            return _fallback("projection returned no finite points for this domain")

        x_range = float(np.ptp(px[finite]))
        y_range = float(np.ptp(py[finite]))
        if not (x_range > 0.0 and y_range > 0.0):
            return _fallback("domain has zero extent in a projected dimension")

        map_aspect = y_range / x_range
    except Exception as exc:
        # Figure sizing must never be the reason an animation fails.
        return _fallback(f"{type(exc).__name__} while projecting the domain")

    axes_width_fraction = _AXES_WIDTH_FRACTION * (_COLORBAR_WIDTH_FRACTION if show_colorbar else 1.0)
    height = _FIG_WIDTH_INCHES * axes_width_fraction * map_aspect / _AXES_HEIGHT_FRACTION
    # ``_MAX_FRAME_PIXELS`` is even, so clamping here survives ``_even_height`` unchanged.
    height = min(max(height, _FIG_WIDTH_INCHES * _MIN_FIG_ASPECT), _MAX_FRAME_PIXELS / _FRAME_DPI)
    return (_FIG_WIDTH_INCHES, _even_height(height))


def _animate(
    plotter,
    config,
    plot_dir: Union[str, Path] = "./",
    file_name: Optional[str] = None,
    centroids: Optional[xr.DataArray] = None,
    object_ids: Optional[xr.DataArray] = None,
) -> Optional[str]:  # pragma: no cover
    """Create an animation from time series data

    Args:
        plotter: The plotter instance driving the animation
        config: Plot configuration (including framerate for animation, default 10 fps)
        plot_dir: Directory to save animation files
        file_name: Name for the output animation file
        centroids: Optional DataArray containing centroid data with dimensions (component, time, ID)
        object_ids: Optional DataArray containing object ID field with integers > 0 for drawing contour outlines
    """
    # Check if PIL is available for image processing
    from .._dependencies import require_dependencies

    require_dependencies(["pillow"], "Animation functionality")

    # Check if ffmpeg is installed
    if shutil.which("ffmpeg") is None:
        warnings.warn(
            "ffmpeg executable not found in system PATH. Cannot create animation.\n"
            "Please install ffmpeg using one of the following methods:\n"
            "  - Linux: sudo apt install ffmpeg (Ubuntu/Debian) or sudo yum install ffmpeg (CentOS/RHEL)\n"
            "  - Conda: conda install -c conda-forge ffmpeg\n"
            "Alternatively, use matplotlib for animation in Jupyter notebooks.",
            stacklevel=2,
        )
        return None

    plot_dir = Path(plot_dir)
    plot_dir.mkdir(exist_ok=True)

    # Use dask's scratch directory for temporary frames
    dask_temp = dask.config.get("temporary-directory", default=None)
    if dask_temp is None:
        dask_temp = tempfile.gettempdir()
    temp_dir = Path(tempfile.mkdtemp(prefix="marex_animate_", dir=dask_temp))

    if not file_name:
        file_name = f"movie_{plotter.da.name}.mp4"
    # Only append the extension when it is missing (avoids ``movie_<name>.mp4.mp4``).
    if not file_name.endswith(".mp4"):
        file_name = f"{file_name}.mp4"

    output_file = plot_dir / file_name

    # Set up plotting parameters
    cmap, norm, clim, var_units, extend = plotter._setup_common_params(config)

    # Resolve dimension/coordinate mappings from the plotter (the validated, data-consistent
    # source of truth) so that a custom-dims plotter is not clobbered by a default config.
    dims_map = getattr(plotter, "dimensions", None) or config.dimensions or {}
    coords_map = getattr(plotter, "coordinates", None) or config.coordinates or {}
    time_dim = dims_map.get("time", "time")
    time_coord = coords_map.get("time", time_dim)
    x_coord = coords_map.get("x", "lon")
    y_coord = coords_map.get("y", "lat")

    plot_params = {
        "cmap": cmap,
        "norm": norm,
        "clim": clim,
        "var_units": var_units,
        "extend": extend,
        "show_colorbar": config.show_colorbar,
        "grid_labels": config.grid_labels,
        "projection": config.projection,
        "x_coord": x_coord,
        "y_coord": y_coord,
    }

    # Size the canvas once, from the domain, and reuse it for every frame. Computed here
    # rather than in ``make_frame`` precisely so that all frames share one value -- deriving
    # it per frame would reintroduce the varying-dimension problem that h264 rejects.
    plot_params["figsize"] = _domain_figsize(
        plot_params["projection"] or ccrs.Robinson(),
        plotter.da[x_coord].values if x_coord in plotter.da.coords else None,
        plotter.da[y_coord].values if y_coord in plotter.da.coords else None,
        bool(config.show_colorbar),
    )

    # Set up grid information if needed. The unstructured plotter always carries
    # ``fpath_tgrid``/``fpath_ckdtree`` attributes (possibly None), so only treat the data as
    # unstructured when a grid path is actually set; otherwise fail early with an informative
    # error rather than an UnboundLocalError deep inside frame rendering.
    grid_info = None
    is_unstructured = hasattr(plotter, "fpath_tgrid") or hasattr(plotter, "fpath_ckdtree")
    tgrid_path = getattr(plotter, "fpath_tgrid", None)
    ckdtree_path = getattr(plotter, "fpath_ckdtree", None)
    if tgrid_path is not None or ckdtree_path is not None:
        grid_info = {
            "type": "unstructured",
            "tgrid_path": tgrid_path,
            "ckdtree_path": ckdtree_path,
            "res": getattr(config, "ckdtree_res", 0.3),
        }
    elif is_unstructured:
        raise VisualisationError(
            "Missing grid specification for unstructured animation",
            details="Unstructured animation requires either triangulation or ckdtree data",
            suggestions=[
                "Provide fpath_tgrid for triangulation-based plotting",
                "Provide fpath_ckdtree for interpolated regular grid plotting",
                "Use specify_grid() to set global grid paths before animating",
            ],
        )

    # For gridded plotters, wrap the periodic longitude boundary (avoids a dateline gap in
    # global animations). ``wrap_lon`` is only defined on the gridded plotter, so its presence
    # doubles as the gridded/unstructured discriminator.
    wrap_fn = getattr(plotter, "wrap_lon", None)

    # Use provided centroids or None if not provided
    centroid_data = centroids

    try:
        # Generate frames using dask for parallel processing
        delayed_tasks = []
        for time_ind in range(len(plotter.da[time_dim])):
            data_slice = plotter.da.isel({time_dim: time_ind})
            if wrap_fn is not None:
                data_slice = wrap_fn(data_slice)

            # Create fresh copy of plot_params for this frame to avoid shared references
            frame_params = plot_params.copy()
            frame_params["time_str"] = str(plotter.da[time_coord].isel({time_dim: time_ind}).dt.strftime("%Y-%m-%d").values)

            # Extract centroids for this time step if available
            if centroid_data is not None:
                try:
                    centroids_time = centroid_data.isel({time_dim: time_ind})
                    frame_params["centroids"] = centroids_time
                except Exception:
                    frame_params["centroids"] = None
            else:
                frame_params["centroids"] = None

            # Extract object IDs for this time step if available
            if object_ids is not None:
                try:
                    object_ids_time = object_ids.isel({time_dim: time_ind})
                    # Wrap the ID field too so it stays shape-consistent with the wrapped data.
                    if wrap_fn is not None and x_coord in object_ids_time.coords:
                        object_ids_time = wrap_fn(object_ids_time)
                    frame_params["object_ids"] = object_ids_time
                except Exception:
                    frame_params["object_ids"] = None
            else:
                frame_params["object_ids"] = None

            delayed_tasks.append(make_frame(data_slice, time_ind, temp_dir, frame_params, grid_info))

        # Process frames in batches to manage memory efficiently
        batch_size = 200
        filenames = []
        for i in range(0, len(delayed_tasks), batch_size):
            batch = delayed_tasks[i : i + batch_size]
            batch_results = dask.compute(*batch)
            filenames.extend(batch_results)
            # Force garbage collection between batches to release memory
            gc.collect()

        filenames = sorted(filenames, key=lambda x: int(x.split("_")[-1].split(".")[0]))

        # Create movie using ffmpeg
        subprocess.run(
            [
                "ffmpeg",
                "-y",
                "-threads",
                "0",
                "-framerate",
                str(config.framerate),
                "-i",
                str(temp_dir / "time_%04d.jpg"),
                "-c:v",
                "libx264",
                "-preset",
                "fast",
                "-crf",
                "22",
                "-pix_fmt",
                "yuv420p",
                "-movflags",
                "+faststart",
                str(output_file),
            ],
            check=True,
        )

        return str(output_file)
    finally:
        # Always clean up the temporary frames directory, even if frame generation or ffmpeg
        # raised (a mid-run error would otherwise leave potentially GBs of JPEGs behind).
        shutil.rmtree(temp_dir, ignore_errors=True)


@dask.delayed
def make_frame(
    data_slice: xr.DataArray,
    time_ind: int,
    temp_dir: Path,
    plot_params: Dict[str, Any],
    grid_info: Optional[Dict[str, Any]] = None,
) -> str:  # pragma: no cover
    """Create a single frame for movies - minimise memory usage with dask

    Args:
        data_slice: The data for this specific frame
        time_ind: Frame index
        temp_dir: Directory for temporary files
        plot_params: Dict containing plotting parameters
        grid_info: Dict containing grid paths and settings for unstructured data
    """
    x_coord = plot_params.get("x_coord", "lon")
    y_coord = plot_params.get("y_coord", "lat")
    projection = plot_params.get("projection") or ccrs.Robinson()

    # Object-oriented Figure/canvas: does not touch the (non-thread-safe) pyplot global state.
    # ``figsize`` is computed once per animation by ``_domain_figsize`` and passed in, so every
    # frame shares it; the fallback keeps a standalone call to this function working.
    fig = Figure(figsize=plot_params.get("figsize") or (_FIG_WIDTH_INCHES, _FIG_WIDTH_INCHES * _DEFAULT_FIG_ASPECT))
    FigureCanvasAgg(fig)
    try:
        ax = fig.add_subplot(1, 1, 1, projection=projection)

        data_slice_np = data_slice.values

        # Set up plot kwargs
        plot_kwargs = {
            "transform": ccrs.PlateCarree(),
            "cmap": plot_params["cmap"],
            "shading": "auto",
        }

        if plot_params.get("norm") is not None:
            plot_kwargs["norm"] = plot_params["norm"]
        elif plot_params.get("clim") is not None:
            plot_kwargs["vmin"] = plot_params["clim"][0]
            plot_kwargs["vmax"] = plot_params["clim"][1]

        im = None
        # Handle different grid types
        if grid_info and grid_info.get("type") == "unstructured":
            try:
                from .unstructured import _load_ckdtree, _load_triangulation
            except ImportError as e:
                raise DependencyError(
                    "Unstructured plotting dependencies missing",
                    details=str(e),
                    suggestions=[
                        "Install plotting dependencies: pip install marEx[plot]",
                        "Check that scipy and matplotlib are available",
                        "Verify unstructured grid support is properly installed",
                    ],
                    context={"missing_dependency": str(e), "plot_type": "unstructured"},
                )

            if grid_info.get("ckdtree_path"):
                # Use cached ckdtree data
                ckdt_data = _load_ckdtree(grid_info["ckdtree_path"], grid_info.get("res", 0.3))
                grid_data = data_slice_np[ckdt_data["indices"]].reshape(ckdt_data["lat"].size, ckdt_data["lon"].size)
                grid_data = np.ma.masked_invalid(grid_data)
                im = ax.pcolormesh(ckdt_data["lon"], ckdt_data["lat"], grid_data, **plot_kwargs)
            elif grid_info.get("tgrid_path"):
                # Use triangulation. tripcolor does not accept the ``shading`` kwarg.
                triang = _load_triangulation(grid_info["tgrid_path"])
                data_masked = np.ma.masked_invalid(data_slice_np)
                tri_kwargs = {k: v for k, v in plot_kwargs.items() if k != "shading"}
                im = ax.tripcolor(triang, data_masked, **tri_kwargs)
        else:
            # Regular grid plotting - use the resolved coordinate names (not hardcoded lat/lon)
            lat = data_slice[y_coord].values
            lon = data_slice[x_coord].values
            im = ax.pcolormesh(lon, lat, data_slice_np, **plot_kwargs)

        time_str = plot_params.get("time_str", f"Frame {time_ind}")
        ax.set_title(time_str, size=12)

        # Plot object ID contours if available
        object_ids_data = plot_params.get("object_ids")
        if object_ids_data is not None:
            try:
                object_ids_np = object_ids_data.values
                # Create binary mask where object IDs > 0
                object_mask = object_ids_np > 0

                if np.any(object_mask):
                    # Handle different grid types for contouring
                    if grid_info and grid_info.get("type") == "unstructured":
                        # For unstructured grids, we need to handle contouring differently
                        # This is more complex and may require interpolation to regular grid
                        pass
                    else:
                        # Regular grid plotting - use the resolved coordinate names
                        lat = data_slice[y_coord].values
                        lon = data_slice[x_coord].values

                        # Draw contours around object boundaries (treating all IDs > 0 the same)
                        ax.contour(
                            lon,
                            lat,
                            object_mask.astype(float),
                            levels=[0.5],
                            colors=["white"],
                            linewidths=1.5,
                            transform=ccrs.PlateCarree(),
                            zorder=6,
                        )
            except Exception:
                # Silently skip object ID contouring if any error occurs
                pass

        # Plot centroids if available
        centroids = plot_params.get("centroids")
        if centroids is not None:
            try:
                # Get unique object IDs present in this frame
                unique_ids = np.unique(data_slice_np)
                unique_ids = unique_ids[unique_ids > 0]  # Remove background (0)

                if len(unique_ids) > 0:
                    # Extract centroid coordinates for present objects
                    # centroids shape: (component, ID) where component 0=lat, 1=lon
                    centroids_np = centroids.values

                    # Find which IDs have valid centroids
                    valid_centroids = []
                    for obj_id in unique_ids:
                        try:
                            # Find ID index in centroids
                            id_idx = np.where(centroids.ID.values == obj_id)[0]
                            if len(id_idx) > 0:
                                idx = id_idx[0]
                                lat_centroid = centroids_np[0, idx]  # component 0 = latitude
                                lon_centroid = centroids_np[1, idx]  # component 1 = longitude

                                # Check if centroid is valid (not NaN)
                                if not (np.isnan(lat_centroid) or np.isnan(lon_centroid)):
                                    valid_centroids.append((lon_centroid, lat_centroid))
                        except (IndexError, KeyError):
                            continue

                    # Plot centroids as scatter points
                    if valid_centroids:
                        centroid_lons, centroid_lats = zip(*valid_centroids)
                        ax.scatter(
                            centroid_lons,
                            centroid_lats,
                            c="black",
                            s=20,
                            marker="o",
                            edgecolors="white",
                            linewidth=1.5,
                            transform=ccrs.PlateCarree(),
                            zorder=5,  # Plot above data but below grid lines
                            alpha=0.8,
                        )
            except Exception:
                # Silently skip centroid plotting if any error occurs
                pass

        if plot_params.get("show_colorbar") and im is not None:
            cb = fig.colorbar(im, shrink=0.6, ax=ax, extend=plot_params.get("extend", "both"))
            if plot_params.get("var_units"):
                cb.ax.set_ylabel(plot_params["var_units"], fontsize=10)
            cb.ax.tick_params(labelsize=10)

        land = cfeature.LAND.with_scale("50m")
        coastlines = cfeature.COASTLINE.with_scale("50m")
        ax.add_feature(land, facecolor="darkgrey", zorder=2)
        ax.add_feature(coastlines, linewidth=0.5, zorder=3)
        ax.gridlines(
            crs=ccrs.PlateCarree(),
            draw_labels=plot_params.get("grid_labels", False),
            linewidth=1,
            color="gray",
            alpha=0.5,
            linestyle="--",
            zorder=4,
        )

        # Save at the shared figure size (no ``bbox_inches="tight"``, which would make frame
        # dimensions content-dependent and break ffmpeg). ``_domain_figsize`` already targets
        # an even pixel height, so the crop below is a safety net rather than the usual path.
        filename = f"time_{time_ind:04d}.jpg"
        frame_path = temp_dir / filename
        fig.savefig(str(frame_path), dpi=_FRAME_DPI)

        image = Image.open(str(frame_path))
        width, height = image.size
        if (width % 2) or (height % 2):
            image = image.crop((0, 0, width - (width % 2), height - (height % 2)))
            image.save(str(frame_path))
        image.close()

        return filename
    finally:
        # Release the figure promptly (belt-and-braces with the OO API, which does not
        # register the figure in any global pyplot state).
        fig.clear()
