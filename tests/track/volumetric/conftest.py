"""Fixtures for volumetric tracking tests: synthetic scenarios turned into merge-loop inputs."""

from types import SimpleNamespace
from typing import Optional

import numpy as np
import pytest

from marEx.track.volumetric.config import VolumetricDims
from marEx.track.volumetric.geometry import build_vertical_geometry
from marEx.track.volumetric.ids import IDAllocator
from marEx.track.volumetric.partition import ScaledCoords
from marEx.track.volumetric.store import VolumeStore
from marEx.track.volumetric.tracker4d import identify_volumes

from .synthetic import make_inputs

VDIMS = VolumetricDims(time="time", z="depth", y="lat", x="lon")


def store_from_field(ids: np.ndarray, cell_volume: np.ndarray) -> VolumeStore:
    """Volume store of every object in a ``(time, z, y, x)`` ID array, one slice at a time."""
    store = VolumeStore()
    for t in range(ids.shape[0]):
        store.update_from_slice(ids[t], cell_volume)
    return store


def build_case(binary: np.ndarray, meta: dict, time_chunk: Optional[int] = None) -> SimpleNamespace:
    """
    Turn a synthetic ``(binary, meta)`` scenario into the inputs of the volumetric merge loop.

    The extremes are identified per timestep with :func:`identify_volumes`, the store holds
    every object's volume, and the allocator continues after the identified IDs.
    """
    kwargs = dict(meta["make_inputs"])
    if time_chunk is not None:
        kwargs["time_chunk"] = time_chunk
    extremes, wet, layer_bounds, cell_areas = make_inputs(binary, **kwargs)
    geom = build_vertical_geometry(
        extremes["depth"].values,
        dict(extremes["depth"].attrs),
        wet,
        zdim="depth",
        ydim="lat",
        xdim="lon",
        lat_deg=extremes["lat"].values,
        lon_deg=extremes["lon"].values,
        layer_bounds=layer_bounds,
        layer_thickness=None,
        cell_volumes=None,
        cell_areas=cell_areas,
        grid_resolution=None,
    )
    allocator = IDAllocator()
    id_field = identify_volumes(extremes, geom, meta["max_gap_m"], False, VDIMS, allocator, materialiser=None)
    ids = id_field.values
    store = store_from_field(ids, geom.cell_volume)
    L_H, L_Z = geom.default_scales
    coords = ScaledCoords(extremes["lat"].values, extremes["lon"].values, geom.midpoint, L_H, L_Z)
    return SimpleNamespace(
        extremes=extremes,
        id_field=id_field,
        ids=ids,
        store=store,
        allocator=allocator,
        geom=geom,
        coords=coords,
        dims=VDIMS,
        cell_volume=geom.cell_volume,
    )


@pytest.fixture
def volumetric_case():
    """Factory: ``volumetric_case(binary, meta, time_chunk=None)`` -> merge-loop inputs."""
    return build_case
