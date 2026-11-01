"""Cell-edge handling for drawing regular lat/lon grids on a cartopy axes."""

from typing import Any

import numpy as np


def _cell_edges(centres: np.ndarray) -> np.ndarray:
    """Cell edges for 1-D cell centres: midpoints, with the outer edges extrapolated half a cell."""
    mid = 0.5 * (centres[:-1] + centres[1:])
    return np.concatenate([[centres[0] - (mid[0] - centres[0])], mid, [centres[-1] + (centres[-1] - mid[-1])]])


def pcolormesh_seam_safe(ax: Any, lons: np.ndarray, lats: np.ndarray, values: np.ndarray, **kwargs: Any) -> Any:
    """``ax.pcolormesh`` that splits any cell straddling the projection's antimeridian.

    A cell centred on the antimeridian (e.g. lon centres 0, 5, ..., 355 under a projection centred
    on 0, where the cell spanning 177.5 to 182.5 is symmetric about 180) gives cartopy a
    degenerate polygon when it cuts the cell at the seam, and the render fails inside shapely.
    Splitting the cell at the seam ourselves, one half on each side, avoids that. Every cell that
    straddles the seam is split, so a straddling cell that rendered before is now drawn as two
    halves meeting at the map's edge, which changes the pixels along the seam. Axes without a
    cartopy projection, and grids with no cell across the seam, are passed straight through.
    """
    params = getattr(getattr(ax, "projection", None), "proj4_params", None)
    lons = np.asarray(lons)
    if not isinstance(params, dict) or lons.ndim != 1 or lons.size < 2 or np.ndim(lats) != 1 or np.size(lats) < 2:
        return ax.pcolormesh(lons, lats, values, **kwargs)

    edges = _cell_edges(lons.astype(float))
    seam = float(params.get("lon_0", 0.0)) + 180.0
    seams = seam + 360.0 * np.arange(np.ceil((edges.min() - seam) / 360.0), np.floor((edges.max() - seam) / 360.0) + 1)

    # Insert each seam that falls strictly inside a cell as an extra edge, duplicating that cell's column.
    cuts = [(int(np.searchsorted(edges, s)) - 1, s) for s in seams if edges.min() < s < edges.max() and s not in edges]
    if not cuts:
        return ax.pcolormesh(lons, lats, values, **kwargs)

    keep = {k: v for k, v in kwargs.items() if k != "shading"}
    new_edges = np.insert(edges, [i + 1 for i, _ in cuts], [s for _, s in cuts])
    cols = np.insert(np.arange(lons.size), [i for i, _ in cuts], [i for i, _ in cuts])
    return ax.pcolormesh(
        new_edges, _cell_edges(np.asarray(lats, dtype=float)), np.asarray(values)[..., cols], shading="flat", **keep
    )
