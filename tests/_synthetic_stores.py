"""Seeded synthetic zarr stores for the larger-than-memory tests, and a driver-side RSS guard.

Both writers build the store one time chunk at a time on dask's threaded scheduler, so writing
a store larger than memory never holds more than a few chunks.
"""

import threading
import time

import dask
import dask.array as da
import numpy as np
import psutil
import xarray as xr

TIME_CHUNK = 30


def _time_chunks(nt, chunk):
    return tuple(min(chunk, nt - s) for s in range(0, nt, chunk))


def _write(ds, path):
    with dask.config.set(scheduler="threads", num_workers=4):
        ds.to_zarr(path, mode="w")


def write_synthetic_sst(path, n_years, ny=256, nx=720, seed=0, time_chunk=TIME_CHUNK):
    """Daily float32 ``sst`` on a regular lat/lon grid; returns its input bytes (``sst.size * 4``).

    A latitude-dependent mean, a hemispheric seasonal cycle and a small trend, plus coherent
    warm and cold patches: 16 x 16-cell blocks each carrying six sinusoids of 23-211 days with
    their own amplitude and phase, so anomalies persist and move between neighbouring blocks.
    Gaussian noise (0.1) keeps the anomaly distribution continuous. Time chunk ``time_chunk``,
    space whole.
    """
    t = np.arange(np.datetime64("2000-01-01"), np.datetime64(f"{2000 + n_years}-01-01"), dtype="datetime64[D]")
    lat = np.linspace(-63.75, 63.75, ny)
    lon = np.arange(nx) * (360.0 / nx)
    rng = np.random.default_rng(seed)
    block, n_modes = 16, 6
    shape = (n_modes, -(-ny // block), -(-nx // block))
    amp, phase = rng.uniform(0.2, 1.0, shape), rng.uniform(0, 2 * np.pi, shape)
    periods = np.array([23.0, 41.0, 67.0, 97.0, 151.0, 211.0])
    amp = np.kron(amp, np.ones((1, block, block)))[:, :ny, :nx]
    phase = np.kron(phase, np.ones((1, block, block)))[:, :ny, :nx]
    clim = 15 + 10 * np.cos(np.deg2rad(lat))[:, None]

    def chunk(block_id=None):
        i = block_id[0]
        tt = np.arange(i * time_chunk, min((i + 1) * time_chunk, t.size), dtype="float64")
        seasonal = 2 * np.sin(2 * np.pi * tt / 365.25)[:, None, None] * np.sign(lat)[None, :, None]
        anomaly = sum(amp[k] * np.sin(2 * np.pi * tt[:, None, None] / periods[k] + phase[k]) for k in range(n_modes))
        noise = np.random.default_rng(1000 + i + 10**6 * seed).normal(0, 0.1, (tt.size, ny, nx))
        return (clim + seasonal + 0.01 * tt[:, None, None] / 365.25 + anomaly + noise).astype("float32")

    arr = da.map_blocks(chunk, chunks=(_time_chunks(t.size, time_chunk), (ny,), (nx,)), dtype="float32")
    ds = xr.Dataset(
        {"sst": (("time", "lat", "lon"), arr)},
        coords={"time": t.astype("datetime64[ns]"), "lat": lat, "lon": lon},
    )
    _write(ds, path)
    return int(arr.size * 4)


def _disc(r):
    yy, xx = np.mgrid[-r : r + 1, -r : r + 1]
    return yy**2 + xx**2 <= r**2


def _paint(frame, row, col, shape):
    h, w = shape.shape
    frame[row - h // 2 : row - h // 2 + h, col - w // 2 : col - w // 2 + w] |= shape


# Planted events live in rows [0, PLANTED_ROWS); background blobs only from BACKGROUND_ROW on, so
# no closing of R_fill <= 4 can join a planted event to anything else.
PLANTED_ROWS, BACKGROUND_ROW, PLANTED_R = 88, 110, 10


def _planted(nt, time_chunk):
    """The planted events, in cell and time-index units, for a store of ``nt`` steps."""
    if nt < 6 * time_chunk:
        raise ValueError(f"nt={nt} is too short to plant events across chunk boundaries (needs >= {6 * time_chunk})")
    mid = (nt // (2 * time_chunk)) * time_chunk  # a chunk boundary near the middle
    lineages = [
        # (name, first step, end step exclusive, row, first column, columns moved per step)
        ("early", time_chunk - 10, time_chunk + 15, 14, 30, 1),
        ("middle", mid - 12, mid + 13, 44, 30, 1),
    ]
    t_merge = mid + 2 * time_chunk + time_chunk // 2  # inside a chunk, after both lineages ended
    merge = {"t_first": t_merge - 8, "t_merge": t_merge, "t_end": t_merge + 6, "row": 76, "cols": (40, 100)}
    return lineages, merge


def write_synthetic_blobs(path, nt, ny, nx, seed=0, time_chunk=TIME_CHUNK):
    """Boolean ``extreme_events`` (time, lat, lon) of moving discs, plus an all-ocean ``mask``.

    Planted events (rows below 88; ``ny >= 128`` and ``nx >= 128`` are required):
    - two lineages of a radius-10 disc moving one column per step, each alive across a time-chunk
      boundary; 317 cells every step (a disc that a Euclidean closing and opening of radius 1-4
      leave unchanged), nothing else within 10 cells of either;
    - two stationary radius-10 discs 60 columns apart that, at ``t_merge``, are replaced by one
      capsule covering both, which lives until ``t_end``: a merge of two parents.
    Background discs of radius 4-8 fill rows >= 110 at random, each moving at up to one cell per
    step for 20-60 steps, about one live blob per 6000 cells.

    Returns ``bool_bytes`` and ``id_bytes`` (the int32 ID field the tracker builds) and the
    planted truth under ``lineages`` and ``merge``, in time-index and cell units.
    """
    if ny < 128 or nx < 128:
        raise ValueError(f"the planted events need ny >= 128 and nx >= 128, got {ny} x {nx}")
    lineages, merge = _planted(nt, time_chunk)
    disc = _disc(PLANTED_R)
    row, (c1, c2) = merge["row"], merge["cols"]
    capsule_h, capsule_w = 2 * PLANTED_R + 1, c2 - c1 + 2 * PLANTED_R + 1
    yy, xx = np.mgrid[-PLANTED_R : PLANTED_R + 1, 0:capsule_w]
    capsule = (yy**2 + (xx - np.clip(xx, PLANTED_R, capsule_w - 1 - PLANTED_R)) ** 2) <= PLANTED_R**2
    assert capsule.shape == (capsule_h, capsule_w)

    rng = np.random.default_rng(seed)
    alive = max(1, (ny - BACKGROUND_ROW) * nx // 6000)
    n_bg = int(np.ceil(nt * alive / 40))
    bg_r = rng.integers(4, 9, n_bg)
    bg_life = rng.integers(20, 61, n_bg)
    bg_t0 = rng.integers(-60, nt, n_bg)
    bg_v = rng.integers(-1, 2, (n_bg, 2))
    lo = np.stack([BACKGROUND_ROW + bg_r, 10 + bg_r], axis=1)
    hi = np.stack([ny - 1 - bg_r, nx - 11 - bg_r], axis=1)
    bg_p0 = rng.integers(lo, hi + 1)
    end = bg_p0 + bg_v * bg_life[:, None]
    bg_v = np.where((end < lo) | (end > hi), 0, bg_v)  # a blob that would leave its band stands still on that axis
    discs = {r: _disc(r) for r in range(4, 9)}

    def chunk(block_id=None):
        i = block_id[0]
        t_lo, t_hi = i * time_chunk, min((i + 1) * time_chunk, nt)
        out = np.zeros((t_hi - t_lo, ny, nx), dtype=bool)
        for t in range(t_lo, t_hi):
            frame = out[t - t_lo]
            for _, first, stop, r0, col0, v in lineages:
                if first <= t < stop:
                    _paint(frame, r0, col0 + v * (t - first), disc)
            if merge["t_first"] <= t < merge["t_merge"]:
                _paint(frame, row, c1, disc)
                _paint(frame, row, c2, disc)
            elif merge["t_merge"] <= t < merge["t_end"]:
                frame[row - PLANTED_R : row + PLANTED_R + 1, c1 - PLANTED_R : c1 - PLANTED_R + capsule_w] |= capsule
            active = np.flatnonzero((bg_t0 <= t) & (t < bg_t0 + bg_life))
            for k in active:
                p = bg_p0[k] + bg_v[k] * (t - bg_t0[k])
                _paint(frame, int(p[0]), int(p[1]), discs[int(bg_r[k])])
        return out

    events = da.map_blocks(chunk, chunks=(_time_chunks(nt, time_chunk), (ny,), (nx,)), dtype=bool)
    time = (np.datetime64("2000-01-01", "D") + np.arange(nt)).astype("datetime64[ns]")
    ds = xr.Dataset(
        {
            "extreme_events": (("time", "lat", "lon"), events),
            "mask": (("lat", "lon"), np.ones((ny, nx), dtype=bool)),
        },
        coords={"time": time, "lat": np.linspace(-63.75, 63.75, ny), "lon": np.arange(nx) * (360.0 / nx)},
    )
    _write(ds, path)
    truth = {
        "lineages": [
            {
                "name": name,
                "t_first": first,
                "t_end": stop,
                "row": r0,
                "cols": [col0 + v * (t - first) for t in range(first, stop)],
                "cells": int(disc.sum()),
                "crosses": next(b for b in range(time_chunk, nt, time_chunk) if first < b < stop),
            }
            for name, first, stop, r0, col0, v in lineages
        ],
        "merge": {**merge, "radius": PLANTED_R},
    }
    return {"bool_bytes": int(events.size), "id_bytes": int(events.size * 4), "truth": truth}


class DriverRSSMonitor:
    """Sample this process's CURRENT resident set size every ``interval`` s while the block runs.

    ``growth`` is ``max(rss) - rss at entry``. Current RSS, not ``ru_maxrss``: a high-water mark
    set before the block would hide any gather inside it that stays below that mark.
    """

    def __init__(self, interval=0.2):
        self.interval = interval
        self._process = psutil.Process()
        self._stop = threading.Event()
        self.samples = []

    def _rss(self):
        return self._process.memory_info().rss

    def _run(self):
        while not self._stop.wait(self.interval):
            self.samples.append(self._rss())

    def __enter__(self):
        self.entry = self._rss()
        self.samples = [self.entry]
        self._thread = threading.Thread(target=self._run, name="driver-rss-monitor", daemon=True)
        self._thread.start()
        self._t0 = time.monotonic()
        return self

    def __exit__(self, *exc):
        self._stop.set()
        self._thread.join()
        self.samples.append(self._rss())
        self.wall = time.monotonic() - self._t0
        self.peak = max(self.samples)
        self.growth = self.peak - self.entry
        return False
