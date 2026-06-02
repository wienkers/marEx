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

from ..exceptions import DependencyError

# Handle optional dependencies for plotting
try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False
    ccrs = None
    cfeature = None

try:
    import matplotlib.pyplot as plt

    HAS_MATPLOTLIB = True
except ImportError:
    HAS_MATPLOTLIB = False
    plt = None

try:
    from PIL import Image

    HAS_PIL = True
except ImportError:
    HAS_PIL = False
    Image = None


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

    output_file = plot_dir / f"{file_name}.mp4"

    # Set up plotting parameters
    cmap, norm, clim, var_units, extend = plotter._setup_common_params(config)

    plot_params = {
        "cmap": cmap,
        "norm": norm,
        "clim": clim,
        "var_units": var_units,
        "extend": extend,
        "show_colorbar": config.show_colorbar,
        "grid_labels": config.grid_labels,
    }

    # Set up grid information if needed
    grid_info = None
    if hasattr(plotter, "fpath_tgrid") or hasattr(plotter, "fpath_ckdtree"):
        grid_info = {
            "type": "unstructured",
            "tgrid_path": getattr(plotter, "fpath_tgrid", None),
            "ckdtree_path": getattr(plotter, "fpath_ckdtree", None),
            "res": 0.3,
        }

    # Generate frames using dask for parallel processing
    delayed_tasks = []
    time_dim = config.dimensions["time"] if config.dimensions else "time"
    time_coord = config.coordinates.get("time", time_dim) if config.coordinates else time_dim

    # Use provided centroids or None if not provided
    centroid_data = centroids

    for time_ind in range(len(plotter.da[time_dim])):
        data_slice = plotter.da.isel({time_dim: time_ind})

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

    # Clean up temporary frames directory
    shutil.rmtree(temp_dir)

    return str(output_file)


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
    # Set up plotting parameters
    plt.rc("text", usetex=False)
    plt.rc("font", family="serif")

    fig = plt.figure(figsize=(7, 5))
    ax = plt.axes(projection=ccrs.Robinson())

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
            # Use triangulation
            triang = _load_triangulation(grid_info["tgrid_path"])
            data_masked = np.ma.masked_invalid(data_slice_np)
            im = ax.tripcolor(triang, data_masked, **plot_kwargs)
    else:
        # Regular grid plotting
        lat = data_slice.lat.values
        lon = data_slice.lon.values
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
                    # Regular grid plotting - use lat/lon coordinates
                    lat = data_slice.lat.values
                    lon = data_slice.lon.values

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

    if plot_params.get("show_colorbar"):
        cb = plt.colorbar(im, shrink=0.6, ax=ax, extend=plot_params.get("extend", "both"))
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

    # Save and process frame
    filename = f"time_{time_ind:04d}.jpg"
    temp_file = temp_dir / f"temp_{filename}"
    fig.savefig(str(temp_file), dpi=300, bbox_inches="tight")
    plt.close(fig)

    # Ensure dimensions are even for video encoding
    image = Image.open(str(temp_file))
    width, height = image.size
    new_width = width - (width % 2)
    new_height = height - (height % 2)
    image = image.resize((new_width, new_height), Image.LANCZOS)

    image.save(str(temp_dir / filename))
    temp_file.unlink()

    return filename
