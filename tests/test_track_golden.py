"""Characterization (golden) snapshot test for the tracker merge/split core.

This test pins the *current* numerical behaviour of the full tracking pipeline
(``marEx.tracker(...).run()``) so that a behaviour-preserving refactor of
``marEx/track.py`` can be verified to leave the outputs bit-for-bit unchanged.

The configuration deliberately enables merging with temporal gap filling and
nearest-neighbour partitioning so that both of the merge/split core routines run:

* ``tracker.split_and_merge_objects`` -- produces the partitioned-merge ledger
  (rows with ``n_parents > 1`` and ``n_children > 1``), exercising both merging
  and splitting of co-tracked objects.
* ``tracker.cluster_rename_objects_and_props`` -- remaps local object IDs into
  the final global event-ID space and recomputes per-event areas/centroids.

Determinism note
----------------
The chosen configuration was verified to be bit-exact reproducible across
*separate processes* (independent ``LocalCluster``s) for every output variable,
including the float32 ``centroid`` field and the row ordering of the merge
ledger. The golden baseline is therefore captured to NetCDF files under
``tests/data/`` rather than recomputed, and the test asserts strict identity.

Baseline files (captured from the pre-refactor code):

* ``tests/data/track_golden_events.nc`` -- the events dataset.
* ``tests/data/track_golden_merges.nc`` -- the merge-events dataset.

The synthetic input is the small deterministic ``extremes_gridded.zarr`` fixture
already used by ``test_gridded_tracking.py``; the full 32-timestep extent is
required because shorter slices do not produce any partitioned merges (12- and
20-timestep slices yield ``total_merges == 0``), so they would not exercise the
merge/split core at all. The full run completes in well under two minutes.
"""

from pathlib import Path

import numpy as np
import xarray as xr

import marEx

# Tracking configuration that exercises both merging and splitting.
# These are exactly the "advanced" settings used by
# test_gridded_tracking.test_advanced_tracking_with_merging.
TRACKING_PARAMS = {
    "area_filter_quartile": 0.5,
    "R_fill": 4,
    "T_fill": 2,
    "allow_merging": True,
    "overlap_threshold": 0.5,
    "nn_partitioning": True,
    "quiet": True,
}

# Standard chunking for tracking (spatial dimensions must be contiguous).
CHUNK_SIZE = {"time": 2, "lat": -1, "lon": -1}


class TestTrackGolden:
    """Golden snapshot test for the tracker merge/split core."""

    @classmethod
    def setup_class(cls):
        """Load the synthetic input data and the golden baseline outputs."""
        data_dir = Path(__file__).parent / "data"

        cls.extremes_data = xr.open_zarr(str(data_dir / "extremes_gridded.zarr"), chunks={}).persist()

        # Golden baselines captured from the pre-refactor code.
        cls.golden_events = xr.open_dataset(str(data_dir / "track_golden_events.nc"))
        cls.golden_merges = xr.open_dataset(str(data_dir / "track_golden_merges.nc"))

    def _run_tracker(self):
        """Run the full tracker with the merge/split configuration."""
        data_bin = self.extremes_data.extreme_events.chunk(CHUNK_SIZE)
        mask = self.extremes_data.mask.where(
            (self.extremes_data.lat < 85) & (self.extremes_data.lat > -90),
            other=False,
        )

        tracker = marEx.tracker(data_bin, mask, **TRACKING_PARAMS)
        events_ds, merges_ds = tracker.run(return_merges=True)

        # Materialise so that comparisons operate on concrete numpy arrays.
        return events_ds.compute(), merges_ds.compute()

    def test_merge_split_core_exercised(self, dask_client_gridded):
        """Sanity check that the chosen configuration runs the merge/split core.

        This guards the golden test itself: if a future change to the synthetic
        data or parameters stopped producing partitioned merges (and hence never
        invoked split_and_merge_objects with real work), the identity assertions
        below would still "pass" trivially while no longer characterising the
        merge/split core. This test fails loudly in that case.
        """
        _, merges_ds = self._run_tracker()

        assert merges_ds.sizes["merge_ID"] > 0, "Configuration produced no partitioned merges"

        n_parents = merges_ds.n_parents.values
        n_children = merges_ds.n_children.values

        assert bool((n_parents > 1).any()), "No merge (n_parents > 1) recorded by split_and_merge_objects"
        assert bool((n_children > 1).any()), "No split (n_children > 1) recorded by split_and_merge_objects"

    def test_events_dataset_identical(self, dask_client_gridded):
        """The full events dataset must be identical to the golden baseline."""
        events_ds, _ = self._run_tracker()

        xr.testing.assert_identical(events_ds, self.golden_events)

    def test_merges_dataset_identical(self, dask_client_gridded):
        """The merge-events dataset must be identical to the golden baseline."""
        _, merges_ds = self._run_tracker()

        xr.testing.assert_identical(merges_ds, self.golden_merges)

    def test_key_output_arrays_identical(self, dask_client_gridded):
        """Raw numerical arrays of the key outputs must match the baseline exactly.

        This complements the dataset-level ``assert_identical`` checks with
        explicit element-wise array comparisons of the load-bearing outputs:
        the ID field, per-event areas and centroids, and the merge ledger.
        """
        events_ds, merges_ds = self._run_tracker()

        # Core tracking result: the labelled ID field.
        np.testing.assert_array_equal(
            events_ds.ID_field.values,
            self.golden_events.ID_field.values,
            err_msg="ID_field array differs from golden baseline",
        )

        # Per-event global IDs and presence mask.
        np.testing.assert_array_equal(
            events_ds.global_ID.values,
            self.golden_events.global_ID.values,
            err_msg="global_ID array differs from golden baseline",
        )
        np.testing.assert_array_equal(
            events_ds.presence.values,
            self.golden_events.presence.values,
            err_msg="presence array differs from golden baseline",
        )

        # Event areas (float32) -- must be bit-exact.
        np.testing.assert_array_equal(
            events_ds.area.values,
            self.golden_events.area.values,
            err_msg="area array differs from golden baseline",
        )

        # Event centroids (float32) -- must be bit-exact.
        np.testing.assert_array_equal(
            events_ds.centroid.values,
            self.golden_events.centroid.values,
            err_msg="centroid array differs from golden baseline",
        )

        # Per-event merge ledger embedded in the events dataset.
        np.testing.assert_array_equal(
            events_ds.merge_ledger.values,
            self.golden_events.merge_ledger.values,
            err_msg="merge_ledger array differs from golden baseline",
        )

        # Partitioned-merge ledger (parent/child IDs, overlaps, counts).
        np.testing.assert_array_equal(
            merges_ds.parent_IDs.values,
            self.golden_merges.parent_IDs.values,
            err_msg="merges parent_IDs array differs from golden baseline",
        )
        np.testing.assert_array_equal(
            merges_ds.child_IDs.values,
            self.golden_merges.child_IDs.values,
            err_msg="merges child_IDs array differs from golden baseline",
        )
        np.testing.assert_array_equal(
            merges_ds.overlap_areas.values,
            self.golden_merges.overlap_areas.values,
            err_msg="merges overlap_areas array differs from golden baseline",
        )
        np.testing.assert_array_equal(
            merges_ds.n_parents.values,
            self.golden_merges.n_parents.values,
            err_msg="merges n_parents array differs from golden baseline",
        )
        np.testing.assert_array_equal(
            merges_ds.n_children.values,
            self.golden_merges.n_children.values,
            err_msg="merges n_children array differs from golden baseline",
        )

    def test_tracking_attributes_identical(self, dask_client_gridded):
        """Reported tracking statistics (attrs) must match the golden baseline."""
        events_ds, _ = self._run_tracker()

        # Compare every attribute present on the golden baseline.
        for key, expected in self.golden_events.attrs.items():
            assert key in events_ds.attrs, f"Missing attribute '{key}' in tracker output"
            actual = events_ds.attrs[key]
            if isinstance(expected, (float, np.floating)):
                np.testing.assert_array_equal(
                    np.asarray(actual),
                    np.asarray(expected),
                    err_msg=f"Attribute '{key}' differs from golden baseline",
                )
            else:
                assert actual == expected, f"Attribute '{key}' differs: {actual!r} != {expected!r}"
