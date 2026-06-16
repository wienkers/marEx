"""Synthetic triangular-mesh scenarios for the unstructured tracker.

A ``nrow x ncol`` grid of 1-degree squares, each split into two triangles (three edge neighbours
per cell). Blobs are rectangles in square space moving east one column per day and carrying a
true lineage label; bridge squares join two blobs for a day range. The truth is known, so a test
can ask whether the tracker fused lineages that were only ever bridged, and whether its output
depends on the input time chunking (it must not).

Ported from the Q8 minimal reproducible examples (2026-09-13); kept dependency-free so the
tracker tests can build inputs without a fixture store.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import xarray as xr

NT = 40


def build_mesh(nrow: int, ncol: int, lat0: float = -20.0, lon0: float = 0.0):
    """Triangle mesh. cell index = 2*(r*ncol + c) + k, k=0 lower-right, k=1 upper-left.

    Returns (neighbours (3, ncells) 1-based int32 with 0 = none, lat, lon, area).
    """
    ncells = 2 * nrow * ncol
    nb = np.zeros((3, ncells), dtype=np.int32)
    lat = np.zeros(ncells, np.float32)
    lon = np.zeros(ncells, np.float32)

    def idx(r, c, k):
        return 2 * (r * ncol + c) + k

    for r in range(nrow):
        for c in range(ncol):
            lower, upper = idx(r, c, 0), idx(r, c, 1)
            nb[0, lower] = upper + 1
            nb[1, lower] = idx(r, c + 1, 1) + 1 if c + 1 < ncol else 0
            nb[2, lower] = idx(r - 1, c, 1) + 1 if r - 1 >= 0 else 0
            nb[0, upper] = lower + 1
            nb[1, upper] = idx(r, c - 1, 0) + 1 if c - 1 >= 0 else 0
            nb[2, upper] = idx(r + 1, c, 0) + 1 if r + 1 < nrow else 0
            lat[lower] = lat0 + r + 1 / 3
            lon[lower] = lon0 + c + 2 / 3
            lat[upper] = lat0 + r + 2 / 3
            lon[upper] = lon0 + c + 1 / 3
    radius = 6.371e6
    sq_area = (np.deg2rad(1.0) * radius) ** 2 * np.cos(np.deg2rad(lat))
    area = (0.5 * sq_area).astype(np.float32)
    return nb, lat, lon, area


def render(nrow: int, ncol: int, nt: int, blobs, bridges):
    """Rasterise blobs and bridges on squares.

    blobs: dicts ``lin, r0, r1, c0, w[, speed, t0, t1]``; rows [r0, r1), cols [c0 + speed*t, +w).
    bridges: dicts ``r0, r1, c0, w, t0, t1[, speed]`` (lineage 0).
    Returns binary (nt, nrow, ncol), lineage (nt, nrow, ncol) int16, bridge_day (nt,) bool.
    """
    binary = np.zeros((nt, nrow, ncol), bool)
    lineage = np.zeros((nt, nrow, ncol), np.int16)
    bridge_day = np.zeros(nt, bool)
    for t in range(nt):
        for b in blobs:
            if not (b.get("t0", 0) <= t < b.get("t1", nt)):
                continue
            c = int(b["c0"] + b.get("speed", 1) * t)
            binary[t, b["r0"] : b["r1"], c : c + b["w"]] = True
            lineage[t, b["r0"] : b["r1"], c : c + b["w"]] = b["lin"]
        for b in bridges:
            if not (b["t0"] <= t < b["t1"]):
                continue
            c = int(b["c0"] + b.get("speed", 1) * t)
            binary[t, b["r0"] : b["r1"], c : c + b["w"]] = True
            bridge_day[t] = True
    return binary, lineage, bridge_day


def squares_to_cells(a: np.ndarray) -> np.ndarray:
    """(nt, nrow, ncol) -> (nt, 2*nrow*ncol); both triangles of a square take its value."""
    nt = a.shape[0]
    return np.repeat(a.reshape(nt, -1), 2, axis=1)


def cells_to_squares(a: np.ndarray, nrow: int, ncol: int) -> np.ndarray:
    """(nt, 2*nrow*ncol) -> (nt, nrow, ncol), the lower triangle of each square."""
    return a.reshape(a.shape[0], nrow, ncol, 2)[..., 0]


def tracker_inputs(binary_sq: np.ndarray, nrow: int, ncol: int, time_chunk: int):
    """Build the (data, mask, neighbours, cell_areas) DataArrays the unstructured tracker takes."""
    nt = binary_sq.shape[0]
    nb, lat, lon, area = build_mesh(nrow, ncol)
    cells = squares_to_cells(binary_sq)
    ncells = cells.shape[1]
    times = pd.date_range("2000-01-01", periods=nt, freq="D")
    spatial = {"lat": ("ncells", lat), "lon": ("ncells", lon)}
    da = xr.DataArray(cells, dims=("time", "ncells"), coords={"time": times, **spatial})
    da = da.chunk({"time": time_chunk, "ncells": -1})
    mask = xr.DataArray(np.ones(ncells, bool), dims="ncells", coords=spatial).chunk({"ncells": -1})
    neighbours = xr.DataArray(nb, dims=("nv", "ncells")).chunk({"nv": -1, "ncells": -1})
    cell_areas = xr.DataArray(area, dims="ncells", coords=spatial).chunk({"ncells": -1})
    return da, mask, neighbours, cell_areas


TRACKER_KWARGS = {
    "R_fill": 0,
    "T_fill": 0,
    "area_filter_absolute": 1,
    "allow_merging": True,
    "overlap_threshold": 0.25,
    "nn_partitioning": True,
    "unstructured_grid": True,
    "dimensions": {"x": "ncells"},
    "coordinates": {"x": "lon", "y": "lat"},
    "regional_mode": False,
    "coordinate_units": "degrees",
    "quiet": True,
}


def scenarios():
    """Hand-built scenarios with known truth (number of distinct lineages = number of events)."""
    s = {}
    # S1: A and B bridged for days 8..29, separate before and after. Truth: 2 events.
    s["S1"] = {
        "nrow": 24,
        "ncol": 60,
        "truth": 2,
        "blobs": [{"lin": 1, "r0": 3, "r1": 9, "c0": 2, "w": 8}, {"lin": 2, "r0": 13, "r1": 19, "c0": 2, "w": 8}],
        "bridges": [{"r0": 9, "r1": 13, "c0": 4, "w": 3, "t0": 8, "t1": 30}],
    }
    # S2: A-B bridged 5..32, B-C bridged 14..32 (a second merge into the complex, later). Truth: 3.
    s["S2"] = {
        "nrow": 30,
        "ncol": 60,
        "truth": 3,
        "blobs": [
            {"lin": 1, "r0": 2, "r1": 7, "c0": 2, "w": 8},
            {"lin": 2, "r0": 11, "r1": 16, "c0": 2, "w": 8},
            {"lin": 3, "r0": 20, "r1": 25, "c0": 2, "w": 8},
        ],
        "bridges": [
            {"r0": 7, "r1": 11, "c0": 4, "w": 3, "t0": 5, "t1": 33},
            {"r0": 16, "r1": 20, "c0": 4, "w": 3, "t0": 14, "t1": 33},
        ],
    }
    # S3: 5 braided lanes, staggered overlapping bridge windows. Truth: 5.
    lanes = [(1, 5), (8, 12), (15, 19), (22, 26), (29, 33)]
    win = [(0, 3, 17), (1, 9, 23), (2, 15, 29), (3, 21, 35), (0, 27, 37)]
    s["S3"] = {
        "nrow": 36,
        "ncol": 60,
        "truth": 5,
        "blobs": [{"lin": i + 1, "r0": a, "r1": b, "c0": 2, "w": 8} for i, (a, b) in enumerate(lanes)],
        "bridges": [{"r0": lanes[i][1], "r1": lanes[i + 1][0], "c0": 4, "w": 3, "t0": t0, "t1": t1} for i, t0, t1 in win],
    }
    # S4: ONE object split by a gap on days 10..19 and rejoined from day 20. Truth: 1 event and no
    # merge. Without consolidation the rejoin is re-partitioned and logged as a merge daily (D-087).
    s["S4"] = {
        "nrow": 24,
        "ncol": 60,
        "truth": 1,
        "blobs": [{"lin": 1, "r0": 3, "r1": 9, "c0": 2, "w": 8}, {"lin": 1, "r0": 13, "r1": 19, "c0": 2, "w": 8}],
        "bridges": [
            {"r0": 9, "r1": 13, "c0": 4, "w": 3, "t0": 0, "t1": 10},
            {"r0": 9, "r1": 13, "c0": 4, "w": 3, "t0": 20, "t1": NT},
        ],
    }
    return s


def random_scenario(seed: int, nt: int = NT):
    """Random braided lanes with random bridge windows; the last 3 days are bridge-free."""
    rng = np.random.default_rng(seed)
    nlanes = int(rng.integers(3, 7))
    ncol = 70
    lanes = []
    r = 2
    for _ in range(nlanes):
        h = int(rng.integers(2, 4))
        lanes.append((r, r + h))
        r += h + int(rng.integers(2, 4))
    nrow = r + 2
    blobs = [
        {"lin": i + 1, "r0": a, "r1": b, "c0": int(rng.integers(1, 6)), "w": int(rng.integers(6, 12)), "speed": 1}
        for i, (a, b) in enumerate(lanes)
    ]
    bridges = []
    for _ in range(int(rng.integers(nlanes, 3 * nlanes))):
        i = int(rng.integers(0, nlanes - 1))
        t0 = int(rng.integers(0, nt - 6))
        t1 = int(min(nt - 3, t0 + rng.integers(3, 25)))
        c0 = max(blobs[i]["c0"], blobs[i + 1]["c0"]) + 1
        bridges.append({"r0": lanes[i][1], "r1": lanes[i + 1][0], "c0": c0, "w": 2, "t0": t0, "t1": t1})
    return {"nrow": nrow, "ncol": ncol, "truth": nlanes, "blobs": blobs, "bridges": bridges}


def same_partition(a: np.ndarray, b: np.ndarray) -> bool:
    """True iff two label fields induce the same partition of the labelled cells (relabelling allowed)."""
    if np.any((a > 0) != (b > 0)):
        return False
    m = a > 0
    pairs = np.unique(np.stack([a[m], b[m]]), axis=1)
    return len(np.unique(pairs[0])) == pairs.shape[1] == len(np.unique(pairs[1]))


def fused_events(idf_sq: np.ndarray, lineage_sq: np.ndarray, bridge_day: np.ndarray):
    """Events holding cells of >= 2 true lineages on days without any bridge: {event: [lineages]}."""
    sets = {}
    for t in np.where(~bridge_day)[0]:
        e, lin = idf_sq[t], lineage_sq[t]
        m = (e > 0) & (lin > 0)
        for eid, li in set(zip(e[m].tolist(), lin[m].tolist())):
            sets.setdefault(eid, set()).add(li)
    return {k: sorted(v) for k, v in sets.items() if len(v) > 1}
