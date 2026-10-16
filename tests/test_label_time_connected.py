"""
``label_time_connected`` must give dask-image's labels and count exactly, with a task graph
linear in the number of time chunks.

``dask_image.ndmeasure.label`` is the reference here, never on the library path: its graph grows
roughly quadratically in the number of chunks (2,037 tasks at 8 time chunks of the field below,
6,517 at 16, and about 82,000 at 64, where the exact count varies with the hash seed).
"""

import sys

import dask
import dask.array as da
import numpy as np
import pytest
from dask_image.ndmeasure import label as dask_image_label

from marEx.track.labelling import label_time_connected

FULL = np.ones((3, 3, 3))


def _reference(data, wrap_x):
    labels, n = dask_image_label(data, structure=FULL, wrap_axes=(2,) if wrap_x else None)
    return np.asarray(labels.compute()), int(n.compute())


def _ours(data, wrap_x):
    labels, n = label_time_connected(data, wrap_x)
    assert labels.chunks == data.chunks
    assert labels.dtype == np.int32
    return np.asarray(labels.compute()), int(n.compute())


def _random_cases():
    cases = []
    for seed in range(24):
        rng = np.random.default_rng(seed)
        nt, ny, nx = int(rng.integers(3, 30)), int(rng.integers(2, 12)), int(rng.integers(2, 16))
        density = float(rng.choice([0.02, 0.1, 0.3, 0.6]))
        field = rng.random((nt, ny, nx)) < density
        cuts = np.sort(rng.choice(np.arange(1, nt), size=min(nt - 1, int(rng.integers(0, 8))), replace=False))
        chunks = tuple(np.diff(np.concatenate(([0], cuts, [nt]))).tolist())
        cases.append(pytest.param(field, chunks, bool(seed % 2), id=f"random{seed}"))
    return cases


def _edge_cases():
    empty = np.zeros((6, 4, 5), dtype=bool)
    full = np.ones((6, 4, 5), dtype=bool)
    # Seam contacts across a time boundary, both diagonals: (t, y, x=-1) touches (t+1, y+-1, x=0).
    corner = np.zeros((6, 5, 7), dtype=bool)
    corner[2, 1, -1] = corner[3, 2, 0] = True  # crosses the boundary between chunks (3,) and (3,)
    corner[2, 4, 0] = corner[3, 3, -1] = True
    # Two objects that only meet through a chain spanning three chunks.
    chain = np.zeros((9, 3, 3), dtype=bool)
    chain[0:9, 0, 0] = True
    chain[4, 2, 2] = chain[5, 1, 1] = True
    return [
        pytest.param(empty, (2, 2, 2), True, id="empty"),
        pytest.param(full, (2, 2, 2), True, id="all_true"),
        pytest.param(full, (6,), False, id="all_true_one_chunk"),
        pytest.param(corner, (3, 3), True, id="seam_corner_wrapped"),
        pytest.param(corner, (3, 3), False, id="seam_corner_regional"),
        pytest.param(corner, (1,) * 6, True, id="seam_corner_time_chunk_1"),
        pytest.param(chain, (3, 3, 3), True, id="chain_across_chunks"),
        pytest.param(chain, (1,) * 9, False, id="chain_time_chunk_1"),
    ]


@pytest.mark.parametrize("field, time_chunks, wrap_x", _random_cases() + _edge_cases())
def test_matches_dask_image(field, time_chunks, wrap_x):
    data = da.from_array(field, chunks=(time_chunks, -1, -1))
    ref_labels, ref_n = _reference(data, wrap_x)
    labels, n = _ours(data, wrap_x)
    assert n == ref_n
    np.testing.assert_array_equal(labels, ref_labels)


def _n_tasks(n_chunks):
    field = da.from_array(np.zeros((4 * n_chunks, 8, 12), dtype=bool), chunks=(4, -1, -1))
    labels, n = label_time_connected(field, wrap_x=True)
    return len(set(dict(labels.__dask_graph__())) | set(dict(n.__dask_graph__())))


def test_task_count_is_linear_in_time_chunks():
    counts = {b: _n_tasks(b) for b in (8, 16, 32, 64)}
    per_chunk = {b: (counts[b] - counts[8]) / (b - 8) for b in (16, 32, 64)}
    # Constant tasks per added chunk: linear, with no term that grows with the chunk count.
    assert len(set(per_chunk.values())) == 1, counts
    assert per_chunk[64] <= 10, counts


def test_resolving_task_receives_no_time_planes():
    """The resolving task must get counts and contacts only, never a chunk's whole label planes.

    Checked on what ``_resolve`` is actually called with, not on graph key names: a plane routed
    through any intermediate task would pass a dependency-name check.
    """
    module = sys.modules[label_time_connected.__module__]
    original, received = module._resolve, []

    def spy(*args):
        received.append(args)
        return original(*args)

    module._resolve = spy
    try:
        field = da.from_array(np.random.default_rng(0).random((16, 8, 12)) < 0.2, chunks=(4, -1, -1))
        labels, n_objects = label_time_connected(field, wrap_x=True)
        dask.compute(labels, n_objects, scheduler="synchronous")
    finally:
        module._resolve = original
    assert len(received) == 1
    leaves = [leaf for arg in received[0] for leaf in (arg if isinstance(arg, (list, tuple)) else [arg])]
    pairs = [leaf for leaf in leaves if isinstance(leaf, np.ndarray)]
    assert pairs and sum(len(p) for p in pairs) > 0, "the field must produce contacts for this check to bite"
    for leaf in leaves:
        if isinstance(leaf, np.ndarray):
            assert leaf.ndim == 2 and leaf.shape[1] == 2 and leaf.dtype == np.int64, (leaf.shape, leaf.dtype)
        else:
            assert isinstance(leaf, (int, np.integer)), type(leaf)


def test_rejects_spatially_chunked_input():
    data = da.zeros((4, 6, 6), dtype=bool, chunks=(2, 3, -1))
    with pytest.raises(ValueError, match="whole in y and x"):
        label_time_connected(data, wrap_x=True)
