"""
marEx.genealogy
===============

Post-processing of marEx tracking output to reconstruct the full merge/split
genealogy of tracked events.

Background
----------
``marEx.tracker`` emits two artefacts:

1. ``events_ds`` — the main tracked-events zarr (``ID_field``, ``area``,
   ``centroid``, ``presence``, ``time_start``, ``time_end``, ``global_ID``).
2. ``genealogy_ds`` — a consolidated sidecar dataset combining:

     * **Partitioned-merge records** (``merge_ID``/``parent_idx``/
       ``child_idx`` dims) logged every time the overlap-based merge
       detector fires, i.e. two or more parents overlap a single
       labelling-child at time *t* with overlap >= ``overlap_threshold``.
     * **Per-timestep adjacency edges** (``edge`` dim) recording every
       pair of distinct event IDs whose labelled regions share at least
       one boundary cell at a given timestep (``adj_time``, ``adj_id_a``,
       ``adj_id_b``, ``adj_boundary_length``).

This module reconstructs the full typed merge/split DAG from those
primitives. Three interaction types are distinguished (terminology
borrowed from the mesoscale-eddy "genealogical evolution" literature,
Li et al. 2016 / Laxenaire 2018 / Tian 2021, adapted for marine
heatwaves):

``PM`` — Partitioned Merge
    >= 2 parents overlap a single labelling child at *t*, all
    overlaps >= ``overlap_threshold``. Parents all continue as
    co-existing (touching) events. Read directly from
    ``/partitioned_merges``.

``AM`` — Absorptive Merge
    Parent A ends at time *t*. At *t*, A is spatially adjacent to parent
    B which persists at *t* + 1. A's overlap with B's labelling-child
    was < ``overlap_threshold``, so A "dies into" B without being
    recorded in ``/partitioned_merges``. Derived from the adjacency
    ledger + per-event ``time_end``.

``PS`` — Partitioned Split
    Events A and B were adjacent (co-tracked while touching) at time
    *t*, both continue to *t* + 1, but adjacency(A, B, t+1) is
    ``False``. Previously-touching events have separated. Derived from
    the adjacency ledger by scanning for disappearing edges.

A single event with spatially disjoint labelled regions under the same
ID is *not* a split — its area and centroid are already computed over
the union by ``track.py``.

Public API
----------

``build_genealogy``
    Build the full typed edge list (PM + AM + PS) and per-event
    statistics, returning an ``xarray.Dataset``.

``load_genealogy``
    Open a previously-built genealogy dataset.

``compute_event_statistics``
    Per-event view of the genealogy dataset as a ``pandas.DataFrame``.

``compute_global_statistics``
    Dataset-wide aggregate counts (PM/AM/PS, genesis/termination, hub
    events, family-size distribution).

``compute_family_statistics``
    Per-connected-component metrics.

``to_networkx``
    Convert to a :class:`networkx.DiGraph` for advanced graph analysis.

``plot_genealogy_timeline``
    Sankey/alluvial-style time-aligned visualisation of the genealogy
    (event lines with thickness ~ area, converging at merges and
    diverging at splits).

``build_adjacency_from_existing``
    Retrofit helper — scan an already-tracked events zarr and build an
    adjacency ledger from ``ID_field`` alone, for datasets that were
    tracked before the genealogy ledger was introduced.
"""

from __future__ import annotations

from pathlib import Path
from typing import TYPE_CHECKING, Optional, Union

import numpy as np
import pandas as pd
import xarray as xr
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components

from .exceptions import create_data_validation_error
from .logging_config import get_logger

if TYPE_CHECKING:  # pragma: no cover
    import matplotlib.axes
    import matplotlib.figure
    import networkx

logger = get_logger(__name__)

PathLike = Union[str, Path]


# ===========================================================================
# Build genealogy
# ===========================================================================


def build_genealogy(
    events: Union[PathLike, xr.Dataset],
    genealogy: Union[PathLike, xr.Dataset],
    *,
    min_boundary_length: int = 1,
    output_path: Optional[PathLike] = None,
) -> xr.Dataset:
    """
    Construct the full typed merge/split DAG from marEx tracking outputs.

    Parameters
    ----------
    events : path or xr.Dataset
        Main tracked-events dataset (``ID_field``, ``area``, ``presence``,
        ``time_start``, ``time_end``, ...). A path is opened with
        :func:`xarray.open_zarr`.
    genealogy : path or xr.Dataset
        Consolidated genealogy dataset produced by
        ``tracker.run(return_genealogy=True)`` — must contain the
        partitioned-merge variables (``parent_IDs``, ``child_IDs``,
        ``overlap_areas``, ``merge_time``, ``n_parents``, ``n_children``)
        and adjacency variables (``adj_time``, ``adj_id_a``, ``adj_id_b``,
        ``adj_boundary_length``). A path is opened with
        :func:`xarray.open_dataset`.
    min_boundary_length : int, default=1
        Filter out adjacency edges below this shared-boundary count.
        ``1`` keeps every recorded edge.
    output_path : path, optional
        If given, write the resulting genealogy dataset to a netCDF file.

    Returns
    -------
    xr.Dataset
        Typed edge list (``edge`` dim) plus per-event metrics
        (``event`` dim). See the module docstring for variable
        definitions.
    """
    events_ds = _as_dataset(events, open_fn=xr.open_zarr)
    genealogy_ds = _as_dataset(genealogy, open_fn=xr.open_dataset)

    _validate_events_dataset(events_ds)
    _validate_genealogy_dataset(genealogy_ds)

    time_coord = _time_coord_name(events_ds)

    # Materialise the per-event metadata we'll reference repeatedly.
    # These are small (n_events,) arrays — safe to compute eagerly.
    event_id = events_ds["ID"].values.astype(np.int32)
    time_start = events_ds["time_start"].values
    time_end = events_ds["time_end"].values
    presence = events_ds["presence"].transpose(time_coord, "ID").values.astype(bool)
    area = events_ds["area"].transpose(time_coord, "ID").values.astype(np.float32)
    times = events_ds[time_coord].values

    id_to_idx = {int(e): i for i, e in enumerate(event_id)}
    n_events = event_id.size

    # -----------------------------------------------------------------
    # Edge construction
    # -----------------------------------------------------------------
    pm_edges = _build_partitioned_merge_edges(genealogy_ds, id_to_idx, times, area)
    adjacency = _extract_adjacency(genealogy_ds, min_boundary_length)
    am_edges = _build_absorptive_merge_edges(adjacency, pm_edges, time_end, id_to_idx, times, area, presence)
    ps_edges = _build_partitioned_split_edges(adjacency, id_to_idx, times, area, presence)

    all_edges = _concat_edges(pm_edges, am_edges, ps_edges)
    logger.info(
        "Genealogy edges: PM=%d, AM=%d, PS=%d", len(pm_edges["source_id"]), len(am_edges["source_id"]), len(ps_edges["source_id"])
    )

    # -----------------------------------------------------------------
    # Per-event statistics
    # -----------------------------------------------------------------
    per_event = _build_per_event_stats(event_id, time_start, time_end, presence, area, all_edges, id_to_idx, times)

    # -----------------------------------------------------------------
    # Assemble output dataset
    # -----------------------------------------------------------------
    out = xr.Dataset(
        data_vars={
            # Per-event
            "event_id": ("event", event_id),
            "time_start": ("event", time_start),
            "time_end": ("event", time_end),
            "lifetime_days": ("event", per_event["lifetime_days"]),
            "max_area": ("event", per_event["max_area"]),
            "mean_area": ("event", per_event["mean_area"]),
            "total_area": ("event", per_event["total_area"]),
            "genesis_type": ("event", per_event["genesis_type"]),
            "fate_type": ("event", per_event["fate_type"]),
            "n_ancestors_PM": ("event", per_event["n_ancestors_PM"]),
            "n_ancestors_AM": ("event", per_event["n_ancestors_AM"]),
            "n_descendants_PM": ("event", per_event["n_descendants_PM"]),
            "n_descendants_PS": ("event", per_event["n_descendants_PS"]),
            "family_id": ("event", per_event["family_id"]),
            # Typed edges
            "edge_type": ("edge", all_edges["edge_type"]),
            "source_id": ("edge", all_edges["source_id"]),
            "target_id": ("edge", all_edges["target_id"]),
            "edge_time": ("edge", all_edges["edge_time"]),
            "overlap_area": ("edge", all_edges["overlap_area"]),
            "source_area": ("edge", all_edges["source_area"]),
            "target_area": ("edge", all_edges["target_area"]),
            "similarity_a": ("edge", all_edges["similarity_a"]),
            "similarity_b": ("edge", all_edges["similarity_b"]),
        },
        coords={"event": np.arange(n_events, dtype=np.int32)},
        attrs={
            "min_boundary_length": int(min_boundary_length),
            "n_events": int(n_events),
            "n_edges_PM": int((all_edges["edge_type"] == "PM").sum()),
            "n_edges_AM": int((all_edges["edge_type"] == "AM").sum()),
            "n_edges_PS": int((all_edges["edge_type"] == "PS").sum()),
        },
    )

    if output_path is not None:
        out.to_netcdf(str(output_path), mode="w")
        logger.info("Wrote genealogy dataset to %s", output_path)

    return out


def load_genealogy(path: PathLike) -> xr.Dataset:
    """Open a genealogy dataset that was previously written by ``build_genealogy``."""
    return xr.open_dataset(str(path))


# ===========================================================================
# Statistics
# ===========================================================================


def compute_event_statistics(genealogy: xr.Dataset) -> pd.DataFrame:
    """Per-event statistics as a pandas DataFrame."""
    cols = [
        "event_id",
        "time_start",
        "time_end",
        "lifetime_days",
        "max_area",
        "mean_area",
        "total_area",
        "genesis_type",
        "fate_type",
        "n_ancestors_PM",
        "n_ancestors_AM",
        "n_descendants_PM",
        "n_descendants_PS",
        "family_id",
    ]
    return genealogy[cols].to_dataframe().reset_index(drop=True)


def compute_global_statistics(genealogy: xr.Dataset) -> dict:
    """Dataset-wide aggregate counts and distributions."""
    n_events = int(genealogy.sizes["event"])
    edge_type = genealogy["edge_type"].values.astype("U2")

    n_PM = int((edge_type == "PM").sum())
    n_AM = int((edge_type == "AM").sum())
    n_PS = int((edge_type == "PS").sum())

    n_desc_PM = genealogy["n_descendants_PM"].values
    n_anc_PM = genealogy["n_ancestors_PM"].values
    n_anc_AM = genealogy["n_ancestors_AM"].values
    n_desc_PS = genealogy["n_descendants_PS"].values

    involved_merge = (n_anc_PM > 0) | (n_anc_AM > 0) | (n_desc_PM > 0)
    involved_split = n_desc_PS > 0
    solitary = (~involved_merge) & (~involved_split)

    family_id = genealogy["family_id"].values
    _, family_sizes = np.unique(family_id, return_counts=True)

    sim_a = genealogy["similarity_a"].values
    sim_b = genealogy["similarity_b"].values
    pm_mask = edge_type == "PM"
    pm_asymmetry = np.abs(sim_a[pm_mask] - sim_b[pm_mask])

    return {
        "n_events": n_events,
        "n_edges_PM": n_PM,
        "n_edges_AM": n_AM,
        "n_edges_PS": n_PS,
        "frac_involved_in_merge": float(involved_merge.mean()) if n_events else 0.0,
        "frac_involved_in_split": float(involved_split.mean()) if n_events else 0.0,
        "frac_solitary": float(solitary.mean()) if n_events else 0.0,
        "n_families": int(family_sizes.size),
        "family_size_mean": float(family_sizes.mean()) if family_sizes.size else 0.0,
        "family_size_max": int(family_sizes.max()) if family_sizes.size else 0,
        "top_hubs_by_merge_degree": _top_k_hubs(genealogy, n_desc_PM + n_anc_PM + n_anc_AM, k=10),
        "pm_asymmetry_mean": float(pm_asymmetry.mean()) if pm_asymmetry.size else float("nan"),
        "pm_asymmetry_median": float(np.median(pm_asymmetry)) if pm_asymmetry.size else float("nan"),
    }


def compute_family_statistics(genealogy: xr.Dataset) -> pd.DataFrame:
    """Per-family (connected-component) aggregate statistics."""
    df = genealogy[["event_id", "family_id", "lifetime_days", "max_area", "total_area"]].to_dataframe().reset_index(drop=True)
    grouped = df.groupby("family_id").agg(
        n_events=("event_id", "count"),
        lifetime_days_total=("lifetime_days", "sum"),
        lifetime_days_max=("lifetime_days", "max"),
        max_area_peak=("max_area", "max"),
        total_area_sum=("total_area", "sum"),
    )
    return grouped.reset_index()


# ===========================================================================
# NetworkX export
# ===========================================================================


def to_networkx(genealogy: xr.Dataset) -> "networkx.DiGraph":
    """
    Convert the typed edge list into a NetworkX directed graph.

    Requires ``networkx`` to be installed. Node attributes include
    ``time_start``, ``time_end``, ``max_area``, ``family_id``; edge
    attributes include ``edge_type``, ``edge_time``, ``overlap_area``,
    ``similarity_a``, ``similarity_b``.
    """
    try:
        import networkx as nx
    except ImportError as exc:  # pragma: no cover
        raise ImportError("networkx is required for marEx.genealogy.to_networkx — install with `pip install networkx`.") from exc

    G = nx.DiGraph()
    ev_ids = genealogy["event_id"].values
    for i, eid in enumerate(ev_ids):
        G.add_node(
            int(eid),
            time_start=genealogy["time_start"].values[i],
            time_end=genealogy["time_end"].values[i],
            lifetime_days=int(genealogy["lifetime_days"].values[i]),
            max_area=float(genealogy["max_area"].values[i]),
            family_id=int(genealogy["family_id"].values[i]),
            genesis_type=int(genealogy["genesis_type"].values[i]),
            fate_type=int(genealogy["fate_type"].values[i]),
        )

    edge_type = genealogy["edge_type"].values.astype("U2")
    src = genealogy["source_id"].values
    tgt = genealogy["target_id"].values
    etime = genealogy["edge_time"].values
    overlap = genealogy["overlap_area"].values
    sim_a = genealogy["similarity_a"].values
    sim_b = genealogy["similarity_b"].values

    for k in range(len(src)):
        G.add_edge(
            int(src[k]),
            int(tgt[k]),
            edge_type=str(edge_type[k]),
            edge_time=etime[k],
            overlap_area=float(overlap[k]) if np.isfinite(overlap[k]) else float("nan"),
            similarity_a=float(sim_a[k]) if np.isfinite(sim_a[k]) else float("nan"),
            similarity_b=float(sim_b[k]) if np.isfinite(sim_b[k]) else float("nan"),
        )
    return G


# ===========================================================================
# Timeline plot
# ===========================================================================


def plot_genealogy_timeline(
    genealogy: xr.Dataset,
    events: xr.Dataset,
    *,
    time_range: Optional[slice] = None,
    seed_ids: Optional[list] = None,
    n_hops: int = 2,
    min_lifetime_days: int = 5,
    area_scale: str = "linear",
    colour_by: str = "family",
    ax: Optional["matplotlib.axes.Axes"] = None,
) -> "matplotlib.figure.Figure":
    """
    Render a Sankey/alluvial-style timeline of the merge/split genealogy.

    Each event is drawn as a filled horizontal band whose x-extent runs
    from ``time_start`` to ``time_end``, whose thickness at each time is
    proportional to ``area(t)`` (``area_scale`` = 'linear' | 'sqrt' |
    'log'), and whose y-position is chosen by a barycentric sweep so that
    merging/splitting events converge/diverge at the correct instants.

    Parameters
    ----------
    genealogy : xr.Dataset
        Output of :func:`build_genealogy`.
    events : xr.Dataset
        Main tracked-events dataset (for ``area`` and ``presence``).
    time_range : slice, optional
        Restrict the x-axis.
    seed_ids : list[int], optional
        Only plot events reachable within ``n_hops`` from these seeds.
    n_hops : int, default=2
        Neighbourhood radius when filtering via ``seed_ids``.
    min_lifetime_days : int, default=5
        Hide events shorter than this (cosmetic cleanup).
    area_scale : {'linear', 'sqrt', 'log'}, default='linear'
        Scale applied to area before mapping to line thickness.
    colour_by : {'family', 'id'}, default='family'
        Colour mapping.
    ax : matplotlib Axes, optional
        Draw onto an existing axes.
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as exc:  # pragma: no cover
        raise ImportError("matplotlib is required for plot_genealogy_timeline") from exc

    time_coord = _time_coord_name(events)
    all_times = events[time_coord].values
    event_ids = genealogy["event_id"].values.astype(np.int32)
    family_id = genealogy["family_id"].values.astype(np.int32)
    time_start = genealogy["time_start"].values
    time_end = genealogy["time_end"].values
    lifetime_days = genealogy["lifetime_days"].values
    area_all = events["area"].transpose(time_coord, "ID").values

    # Filter by lifetime / time window / seed neighbourhood.
    keep = lifetime_days >= int(min_lifetime_days)
    if time_range is not None:
        t_lo = np.datetime64(time_range.start) if time_range.start is not None else all_times[0]
        t_hi = np.datetime64(time_range.stop) if time_range.stop is not None else all_times[-1]
        keep &= (time_end >= t_lo) & (time_start <= t_hi)
    if seed_ids is not None:
        keep &= _neighbourhood_mask(genealogy, seed_ids, n_hops)

    idx = np.where(keep)[0]
    if idx.size == 0:
        raise ValueError("No events remain after filtering.")

    # Group by family, then order within family by time_start.
    order = sorted(idx, key=lambda i: (family_id[i], time_start[i]))
    y_positions = {int(event_ids[i]): rank for rank, i in enumerate(order)}

    # Single-pass barycentric relaxation using overlap_time edges as springs.
    src = genealogy["source_id"].values.astype(np.int32)
    tgt = genealogy["target_id"].values.astype(np.int32)
    for _ in range(8):
        new_y = dict(y_positions)
        accum: dict = {i: [y_positions[i]] for i in y_positions}
        for s, t in zip(src, tgt):
            if int(s) in y_positions and int(t) in y_positions:
                accum[int(s)].append(y_positions[int(t)])
                accum[int(t)].append(y_positions[int(s)])
        for node, ys in accum.items():
            new_y[node] = float(np.mean(ys))
        # Rank-normalise to keep y-spacing uniform.
        ranked = sorted(new_y, key=new_y.get)
        y_positions = {node: rank for rank, node in enumerate(ranked)}

    # Area scaling helper.
    def _scale(a: np.ndarray) -> np.ndarray:
        if area_scale == "linear":
            return a
        if area_scale == "sqrt":
            return np.sqrt(np.clip(a, 0, None))
        if area_scale == "log":
            return np.log1p(np.clip(a, 0, None))
        raise ValueError(f"unknown area_scale: {area_scale}")

    max_area_all = np.nanmax(_scale(np.where(np.isnan(area_all), 0, area_all)))
    thickness_norm = 0.4 / max_area_all if max_area_all > 0 else 0.0

    if ax is None:
        fig, ax = plt.subplots(figsize=(16, max(4, len(idx) * 0.12)))
    else:
        fig = ax.figure

    cmap = plt.cm.tab20
    for i in idx:
        eid = int(event_ids[i])
        y = y_positions[eid]
        event_area = _scale(np.nan_to_num(area_all[:, list(events["ID"].values).index(eid)], nan=0.0))
        half = event_area * thickness_norm
        mask = event_area > 0
        if not np.any(mask):
            continue
        colour_idx = family_id[i] if colour_by == "family" else eid
        colour = cmap(int(colour_idx) % 20)
        ax.fill_between(all_times[mask], y - half[mask], y + half[mask], color=colour, linewidth=0, alpha=0.85)

    ax.set_xlabel("time")
    ax.set_ylabel("event (ordered by family + barycentric sweep)")
    ax.set_yticks([])
    ax.set_title(f"marEx genealogy — {len(idx)} events")
    return fig


# ===========================================================================
# Retrofit helper
# ===========================================================================


def build_adjacency_from_existing(
    events_zarr_path: PathLike,
    *,
    legacy_merges_nc: Optional[PathLike] = None,
    output_path: Optional[PathLike] = None,
) -> xr.Dataset:
    """
    Retrofit an adjacency ledger onto a tracked-events zarr that predates
    the consolidated genealogy output.

    Scans ``ID_field`` in the existing zarr and applies the same
    per-timestep adjacency logic used by ``marEx.tracker`` in the
    recompute-properties pass. If ``legacy_merges_nc`` is provided, its
    contents are merged into the result so downstream consumers see a
    complete consolidated genealogy dataset.

    Parameters
    ----------
    events_zarr_path : path
        Path to the main events zarr store.
    legacy_merges_nc : path, optional
        Path to a legacy ``*_merges.nc`` file (as emitted by older
        marEx runs) whose variables will be merged into the result.
    output_path : path, optional
        If given, write the resulting dataset to netCDF.

    Returns
    -------
    xr.Dataset
        Consolidated genealogy dataset matching the schema emitted by
        ``tracker.run(return_genealogy=True)``.
    """
    ds = xr.open_zarr(str(events_zarr_path))
    time_coord = _time_coord_name(ds)

    id_field = ds["ID_field"]
    presence = ds["presence"].transpose(time_coord, "ID").values.astype(bool)
    event_ids = ds["ID"].values.astype(np.int32)
    times = ds[time_coord].values

    is_unstructured = "ncells" in id_field.dims
    ydim = None if is_unstructured else [d for d in id_field.dims if d not in (time_coord, "lon", "longitude", "x")]
    ydim = ydim[0] if ydim else None

    adj_times = []
    adj_ids_a = []
    adj_ids_b = []
    adj_counts = []

    n_time = id_field.sizes[time_coord]
    for t_idx in range(n_time):
        slice_data = id_field.isel({time_coord: t_idx}).values
        pairs_a, pairs_b = (
            _slice_adjacency_structured(slice_data)
            if not is_unstructured
            else (np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32))
        )
        if pairs_a.size == 0:
            continue
        id_small = np.minimum(pairs_a, pairs_b).astype(np.int32)
        id_large = np.maximum(pairs_a, pairs_b).astype(np.int32)
        uniq, counts = np.unique(np.stack([id_small, id_large], axis=1), axis=0, return_counts=True)
        if uniq.shape[0] == 0:
            continue
        adj_times.append(np.full(uniq.shape[0], times[t_idx], dtype=times.dtype))
        adj_ids_a.append(uniq[:, 0])
        adj_ids_b.append(uniq[:, 1])
        adj_counts.append(counts.astype(np.int32))

    if adj_times:
        adjacency_vars = {
            "adj_time": ("edge", np.concatenate(adj_times)),
            "adj_id_a": ("edge", np.concatenate(adj_ids_a).astype(np.int32)),
            "adj_id_b": ("edge", np.concatenate(adj_ids_b).astype(np.int32)),
            "adj_boundary_length": ("edge", np.concatenate(adj_counts).astype(np.int32)),
        }
    else:
        adjacency_vars = {
            "adj_time": ("edge", np.empty(0, dtype=times.dtype)),
            "adj_id_a": ("edge", np.empty(0, dtype=np.int32)),
            "adj_id_b": ("edge", np.empty(0, dtype=np.int32)),
            "adj_boundary_length": ("edge", np.empty(0, dtype=np.int32)),
        }

    result = xr.Dataset(adjacency_vars, attrs={"retrofitted": 1})

    if legacy_merges_nc is not None:
        legacy = xr.open_dataset(str(legacy_merges_nc))
        result = xr.merge([legacy, result], combine_attrs="drop_conflicts")

    if output_path is not None:
        result.to_netcdf(str(output_path), mode="w")

    # Touch event_ids/presence to silence unused-variable warnings even though
    # they are loaded for side-effect validation above.
    del event_ids, presence
    return result


# ===========================================================================
# Internal helpers
# ===========================================================================


def _as_dataset(obj, open_fn) -> xr.Dataset:
    if isinstance(obj, xr.Dataset):
        return obj
    return open_fn(str(obj))


def _time_coord_name(ds: xr.Dataset) -> str:
    for cand in ("time", "T", "t"):
        if cand in ds.dims:
            return cand
    raise create_data_validation_error(
        "Could not identify the time dimension on the events dataset",
        data_info={"dims": list(ds.dims)},
        suggestions=["Rename the time dimension to 'time' or pass a dataset with a recognised time dim."],
    )


def _validate_events_dataset(ds: xr.Dataset) -> None:
    required = {"ID_field", "area", "presence", "time_start", "time_end"}
    missing = required - set(ds.data_vars)
    if missing:
        raise create_data_validation_error(
            "Events dataset is missing required variables",
            data_info={"missing": sorted(missing), "present": sorted(ds.data_vars)},
            suggestions=["Pass the main zarr produced by marEx.tracker.run()"],
        )


def _validate_genealogy_dataset(ds: xr.Dataset) -> None:
    required_pm = {"parent_IDs", "child_IDs", "overlap_areas", "merge_time", "n_parents", "n_children"}
    required_adj = {"adj_time", "adj_id_a", "adj_id_b", "adj_boundary_length"}
    missing_pm = required_pm - set(ds.data_vars)
    missing_adj = required_adj - set(ds.data_vars)
    if missing_pm and missing_adj:
        raise create_data_validation_error(
            "Genealogy dataset is missing both partitioned-merge and adjacency variables",
            data_info={"missing_pm": sorted(missing_pm), "missing_adj": sorted(missing_adj)},
            suggestions=["Pass the genealogy dataset from tracker.run(return_genealogy=True)"],
        )


def _build_partitioned_merge_edges(
    genealogy_ds: xr.Dataset,
    id_to_idx: dict,
    times: np.ndarray,
    area: np.ndarray,
) -> dict:
    if "merge_ID" not in genealogy_ds.dims:
        return _empty_edges()

    parent_IDs = genealogy_ds["parent_IDs"].values
    child_IDs = genealogy_ds["child_IDs"].values
    overlap_areas = genealogy_ds["overlap_areas"].values.astype(np.float32)
    merge_time = genealogy_ds["merge_time"].values

    source_id = []
    target_id = []
    edge_time = []
    overlap = []
    source_area = []
    target_area = []

    time_index = pd.Series(np.arange(times.size, dtype=np.int64), index=pd.Index(times))

    for merge_idx in range(parent_IDs.shape[0]):
        mt = merge_time[merge_idx]
        t_pos_arr = time_index.get(mt)
        if t_pos_arr is None:
            continue
        t_pos = int(t_pos_arr) if np.isscalar(t_pos_arr) else int(t_pos_arr[0])
        for p_slot in range(parent_IDs.shape[1]):
            p = int(parent_IDs[merge_idx, p_slot])
            if p < 0:
                continue
            for c_slot in range(child_IDs.shape[1]):
                c = int(child_IDs[merge_idx, c_slot])
                if c < 0:
                    continue
                source_id.append(p)
                target_id.append(c)
                edge_time.append(mt)
                overlap.append(float(overlap_areas[merge_idx, p_slot]))
                source_area.append(_area_at(area, p, t_pos, id_to_idx))
                target_area.append(_area_at(area, c, t_pos, id_to_idx))

    source_id = np.asarray(source_id, dtype=np.int32)
    target_id = np.asarray(target_id, dtype=np.int32)
    edge_time = np.asarray(edge_time, dtype=times.dtype if times.size else "datetime64[ns]")
    overlap = np.asarray(overlap, dtype=np.float32)
    source_area = np.asarray(source_area, dtype=np.float32)
    target_area = np.asarray(target_area, dtype=np.float32)
    sim_a = np.where(source_area > 0, overlap / source_area, np.nan).astype(np.float32)
    sim_b = np.where(target_area > 0, overlap / target_area, np.nan).astype(np.float32)

    return {
        "edge_type": np.full(source_id.shape, "PM", dtype="U2"),
        "source_id": source_id,
        "target_id": target_id,
        "edge_time": edge_time,
        "overlap_area": overlap,
        "source_area": source_area,
        "target_area": target_area,
        "similarity_a": sim_a,
        "similarity_b": sim_b,
    }


def _extract_adjacency(genealogy_ds: xr.Dataset, min_boundary_length: int) -> dict:
    if "edge" not in genealogy_ds.dims:
        return {
            "time": np.empty(0, dtype="datetime64[ns]"),
            "a": np.empty(0, dtype=np.int32),
            "b": np.empty(0, dtype=np.int32),
            "len": np.empty(0, dtype=np.int32),
        }
    a = genealogy_ds["adj_id_a"].values.astype(np.int32)
    b = genealogy_ds["adj_id_b"].values.astype(np.int32)
    t = genealogy_ds["adj_time"].values
    length = genealogy_ds["adj_boundary_length"].values.astype(np.int32)
    mask = length >= int(min_boundary_length)
    return {"time": t[mask], "a": a[mask], "b": b[mask], "len": length[mask]}


def _build_absorptive_merge_edges(
    adjacency: dict,
    pm_edges: dict,
    time_end: np.ndarray,
    id_to_idx: dict,
    times: np.ndarray,
    area: np.ndarray,
    presence: np.ndarray,
) -> dict:
    _ = presence  # reserved for future filtering; signature kept for consistency with caller
    a = adjacency["a"]
    b = adjacency["b"]
    t = adjacency["time"]
    lengths = adjacency["len"]
    if a.size == 0:
        return _empty_edges()

    # Map each adjacency row to the per-event time_end of the a/b endpoints.
    a_idx = np.array([id_to_idx.get(int(x), -1) for x in a], dtype=np.int64)
    b_idx = np.array([id_to_idx.get(int(x), -1) for x in b], dtype=np.int64)
    valid = (a_idx >= 0) & (b_idx >= 0)
    a_idx, b_idx = a_idx[valid], b_idx[valid]
    a, b, t, lengths = a[valid], b[valid], t[valid], lengths[valid]

    t_end_a = time_end[a_idx]
    t_end_b = time_end[b_idx]

    # Exclude rows that coincide with PM events (the parent's end at this
    # timestep is already explained).
    pm_keys = {(int(s), int(tm)) for s, tm in zip(pm_edges["source_id"], pm_edges["edge_time"])}

    # Time index lookup for areas.
    time_to_idx = pd.Series(np.arange(times.size, dtype=np.int64), index=pd.Index(times))

    src, tgt, etime, sarea, tarea, ov = [], [], [], [], [], []
    for row_idx in range(a.size):
        ca = int(a[row_idx])
        cb = int(b[row_idx])
        row_t = t[row_idx]
        # Case 1: A ends at this timestep, touching B which persists.
        if t_end_a[row_idx] == row_t and t_end_b[row_idx] > row_t and (ca, row_t) not in pm_keys:
            t_pos = int(time_to_idx.get(row_t, -1))
            if t_pos < 0:
                continue
            src.append(ca)
            tgt.append(cb)
            etime.append(row_t)
            sa = _area_at(area, ca, t_pos, id_to_idx)
            ta = _area_at(area, cb, t_pos, id_to_idx)
            sarea.append(sa)
            tarea.append(ta)
            ov.append(float(lengths[row_idx]))
        # Case 2: B ends at this timestep, touching A which persists.
        elif t_end_b[row_idx] == row_t and t_end_a[row_idx] > row_t and (cb, row_t) not in pm_keys:
            t_pos = int(time_to_idx.get(row_t, -1))
            if t_pos < 0:
                continue
            src.append(cb)
            tgt.append(ca)
            etime.append(row_t)
            sa = _area_at(area, cb, t_pos, id_to_idx)
            ta = _area_at(area, ca, t_pos, id_to_idx)
            sarea.append(sa)
            tarea.append(ta)
            ov.append(float(lengths[row_idx]))

    if not src:
        return _empty_edges()

    src = np.asarray(src, dtype=np.int32)
    tgt = np.asarray(tgt, dtype=np.int32)
    etime_arr = np.asarray(etime, dtype=times.dtype if times.size else "datetime64[ns]")
    sarea_arr = np.asarray(sarea, dtype=np.float32)
    tarea_arr = np.asarray(tarea, dtype=np.float32)
    ov_arr = np.asarray(ov, dtype=np.float32)
    sim_a = np.where(sarea_arr > 0, ov_arr / sarea_arr, np.nan).astype(np.float32)
    sim_b = np.where(tarea_arr > 0, ov_arr / tarea_arr, np.nan).astype(np.float32)

    return {
        "edge_type": np.full(src.shape, "AM", dtype="U2"),
        "source_id": src,
        "target_id": tgt,
        "edge_time": etime_arr,
        "overlap_area": ov_arr,
        "source_area": sarea_arr,
        "target_area": tarea_arr,
        "similarity_a": sim_a,
        "similarity_b": sim_b,
    }


def _build_partitioned_split_edges(
    adjacency: dict,
    id_to_idx: dict,
    times: np.ndarray,
    area: np.ndarray,
    presence: np.ndarray,
) -> dict:
    a = adjacency["a"]
    b = adjacency["b"]
    t = adjacency["time"]
    if a.size == 0 or times.size < 2:
        return _empty_edges()

    time_to_idx = pd.Series(np.arange(times.size, dtype=np.int64), index=pd.Index(times))
    present_pairs = {(int(ai), int(bi), int(time_to_idx.get(ti, -1))) for ai, bi, ti in zip(a, b, t)}

    src, tgt, etime = [], [], []
    for ai, bi, ti in zip(a, b, t):
        t_pos = int(time_to_idx.get(ti, -1))
        if t_pos < 0 or t_pos + 1 >= times.size:
            continue
        # Pair present at t but NOT at t+1, and both endpoints still present.
        if (int(ai), int(bi), t_pos + 1) in present_pairs:
            continue
        a_row = id_to_idx.get(int(ai), -1)
        b_row = id_to_idx.get(int(bi), -1)
        if a_row < 0 or b_row < 0:
            continue
        if not (presence[t_pos + 1, a_row] and presence[t_pos + 1, b_row]):
            continue
        src.append(int(ai))
        tgt.append(int(bi))
        etime.append(times[t_pos + 1])

    if not src:
        return _empty_edges()

    src = np.asarray(src, dtype=np.int32)
    tgt = np.asarray(tgt, dtype=np.int32)
    etime_arr = np.asarray(etime, dtype=times.dtype)

    # Source/target area at edge_time.
    sarea = np.array(
        [_area_at(area, int(s), int(time_to_idx.get(t, -1)), id_to_idx) for s, t in zip(src, etime_arr)], dtype=np.float32
    )
    tarea = np.array(
        [_area_at(area, int(tg), int(time_to_idx.get(t, -1)), id_to_idx) for tg, t in zip(tgt, etime_arr)], dtype=np.float32
    )
    nan = np.full(src.shape, np.nan, dtype=np.float32)

    return {
        "edge_type": np.full(src.shape, "PS", dtype="U2"),
        "source_id": src,
        "target_id": tgt,
        "edge_time": etime_arr,
        "overlap_area": nan.copy(),
        "source_area": sarea,
        "target_area": tarea,
        "similarity_a": nan.copy(),
        "similarity_b": nan.copy(),
    }


def _empty_edges() -> dict:
    return {
        "edge_type": np.empty(0, dtype="U2"),
        "source_id": np.empty(0, dtype=np.int32),
        "target_id": np.empty(0, dtype=np.int32),
        "edge_time": np.empty(0, dtype="datetime64[ns]"),
        "overlap_area": np.empty(0, dtype=np.float32),
        "source_area": np.empty(0, dtype=np.float32),
        "target_area": np.empty(0, dtype=np.float32),
        "similarity_a": np.empty(0, dtype=np.float32),
        "similarity_b": np.empty(0, dtype=np.float32),
    }


def _concat_edges(*edge_dicts: dict) -> dict:
    if not edge_dicts:
        return _empty_edges()
    keys = edge_dicts[0].keys()
    return {k: np.concatenate([d[k] for d in edge_dicts]) for k in keys}


def _area_at(area: np.ndarray, event_id: int, t_pos: int, id_to_idx: dict) -> float:
    row = id_to_idx.get(int(event_id), -1)
    if row < 0 or t_pos < 0 or t_pos >= area.shape[0]:
        return float("nan")
    val = area[t_pos, row]
    return float(val) if np.isfinite(val) else float("nan")


def _build_per_event_stats(
    event_id: np.ndarray,
    time_start: np.ndarray,
    time_end: np.ndarray,
    presence: np.ndarray,
    area: np.ndarray,
    all_edges: dict,
    id_to_idx: dict,
    times: np.ndarray,
) -> dict:
    n = event_id.size
    lifetime = ((time_end - time_start) / np.timedelta64(1, "D")).astype(np.int32) + 1

    with np.errstate(invalid="ignore"):
        max_area = np.nanmax(np.where(presence, area, np.nan), axis=0).astype(np.float32)
        mean_area = np.nanmean(np.where(presence, area, np.nan), axis=0).astype(np.float32)
        total_area = np.nansum(np.where(presence, area, 0.0), axis=0).astype(np.float64)

    n_anc_PM = np.zeros(n, dtype=np.int32)
    n_anc_AM = np.zeros(n, dtype=np.int32)
    n_desc_PM = np.zeros(n, dtype=np.int32)
    n_desc_PS = np.zeros(n, dtype=np.int32)

    edge_type = all_edges["edge_type"]
    src = all_edges["source_id"]
    tgt = all_edges["target_id"]

    for et, s, t in zip(edge_type, src, tgt):
        s_row = id_to_idx.get(int(s), -1)
        t_row = id_to_idx.get(int(t), -1)
        if et == "PM":
            if t_row >= 0:
                n_anc_PM[t_row] += 1
            if s_row >= 0:
                n_desc_PM[s_row] += 1
        elif et == "AM":
            if t_row >= 0:
                # absorber does not gain an "ancestor"; absorbed has exactly one target.
                pass
            if s_row >= 0:
                n_anc_AM[s_row] += 1
        elif et == "PS":
            if s_row >= 0:
                n_desc_PS[s_row] += 1
            if t_row >= 0:
                n_desc_PS[t_row] += 1

    # Genesis / fate classification.
    genesis_type = np.zeros(n, dtype=np.int8)  # 0 = birth
    fate_type = np.zeros(n, dtype=np.int8)  # 0 = death
    genesis_type[n_anc_PM > 0] = 1  # PM-spawn
    genesis_type[n_anc_AM > 0] = 1  # absorbed parent — still "birth" (no ancestor), AM doesn't create genesis
    fate_type[n_desc_PM > 0] = 1  # contributed to a PM
    fate_type[n_anc_AM > 0] = 1  # absorbed → fate=absorbed (use 1 to mark non-death)
    fate_type[n_desc_PS > 0] = 2  # split

    # Family clustering via connected components over the undirected edge set.
    if src.size > 0:
        rows = np.concatenate(
            [
                np.array([id_to_idx.get(int(x), -1) for x in src], dtype=np.int64),
                np.array([id_to_idx.get(int(x), -1) for x in tgt], dtype=np.int64),
            ]
        )
        cols = np.concatenate(
            [
                np.array([id_to_idx.get(int(x), -1) for x in tgt], dtype=np.int64),
                np.array([id_to_idx.get(int(x), -1) for x in src], dtype=np.int64),
            ]
        )
        mask = (rows >= 0) & (cols >= 0)
        rows, cols = rows[mask], cols[mask]
        data = np.ones(rows.size, dtype=np.bool_)
        graph = csr_matrix((data, (rows, cols)), shape=(n, n))
        _, family_id = connected_components(csgraph=graph, directed=False, return_labels=True)
        family_id = family_id.astype(np.int32)
    else:
        family_id = np.arange(n, dtype=np.int32)

    del times  # unused
    return {
        "lifetime_days": lifetime,
        "max_area": np.nan_to_num(max_area, nan=0.0),
        "mean_area": np.nan_to_num(mean_area, nan=0.0),
        "total_area": np.nan_to_num(total_area, nan=0.0),
        "genesis_type": genesis_type,
        "fate_type": fate_type,
        "n_ancestors_PM": n_anc_PM,
        "n_ancestors_AM": n_anc_AM,
        "n_descendants_PM": n_desc_PM,
        "n_descendants_PS": n_desc_PS,
        "family_id": family_id,
    }


def _top_k_hubs(genealogy: xr.Dataset, degree: np.ndarray, k: int = 10) -> list:
    order = np.argsort(-degree)[:k]
    return [
        {
            "event_id": int(genealogy["event_id"].values[i]),
            "degree": int(degree[i]),
            "family_id": int(genealogy["family_id"].values[i]),
        }
        for i in order
        if degree[i] > 0
    ]


def _neighbourhood_mask(genealogy: xr.Dataset, seed_ids: list, n_hops: int) -> np.ndarray:
    event_id = genealogy["event_id"].values
    src = genealogy["source_id"].values
    tgt = genealogy["target_id"].values
    id_to_idx = {int(e): i for i, e in enumerate(event_id)}

    current = {int(s) for s in seed_ids if int(s) in id_to_idx}
    reached = set(current)
    for _ in range(max(0, int(n_hops))):
        next_set = set()
        for s, t in zip(src, tgt):
            if int(s) in current:
                next_set.add(int(t))
            if int(t) in current:
                next_set.add(int(s))
        current = next_set - reached
        reached |= next_set
    mask = np.zeros(event_id.size, dtype=bool)
    for eid in reached:
        i = id_to_idx.get(eid, -1)
        if i >= 0:
            mask[i] = True
    return mask


def _slice_adjacency_structured(slice_data: np.ndarray):
    a_lists = []
    b_lists = []
    left = slice_data[:, :-1]
    right = slice_data[:, 1:]
    m = (left > 0) & (right > 0) & (left != right)
    if np.any(m):
        a_lists.append(left[m])
        b_lists.append(right[m])
    top = slice_data[:-1, :]
    bottom = slice_data[1:, :]
    m = (top > 0) & (bottom > 0) & (top != bottom)
    if np.any(m):
        a_lists.append(top[m])
        b_lists.append(bottom[m])
    if slice_data.shape[1] > 1:
        wleft = slice_data[:, -1]
        wright = slice_data[:, 0]
        m = (wleft > 0) & (wright > 0) & (wleft != wright)
        if np.any(m):
            a_lists.append(wleft[m])
            b_lists.append(wright[m])
    if a_lists:
        return np.concatenate(a_lists), np.concatenate(b_lists)
    return np.empty(0, dtype=np.int32), np.empty(0, dtype=np.int32)
