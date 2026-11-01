.. _validation-guide:

==============================
Validation and Reproducibility
==============================

marEx is checked in three ways: its outputs are compared with stored reference outputs at zero
tolerance, the same answer is required from different chunkings and compute modes, and the claims
that need real scale are measured on real data (see :ref:`performance-measured`). The test suite in
``tests/`` contains about 2,100 tests.

Reference-Output Tests
======================

``tests/detect/test_pipeline_golden.py`` pins :func:`marEx.preprocess_data` on two configurations of a
deterministic gridded fixture, using the last seven years (2555 daily steps):

* ``detrend_harmonic`` with ``global_percentile``, which exercises the one-dimensional histogram path.
* ``shifting_baseline`` with ``seasonal_percentile``, which exercises the per-day-of-year path.

Both use approximate percentiles, and the stored reference outputs are Zarr stores in
``tests/data/``. Every comparison is made at zero tolerance, except the thresholds of the first
configuration, which are compared with an absolute tolerance of ``2e-14`` (float64 round-off).
Integer and label outputs have no tolerance at all.

The reference runs fix ``precision=0.01`` and a pinned range, so they are independent of the default
bin choice. ``test_legacy_bins_reproduce_the_goldens`` reproduces the references with the legacy bin
layout, so a failure there points at the histogram code. ``tests/track/test_tracker_outputs.py``
does the same for the tracker, against a stored event dataset and merge dataset. Reference outputs
are regenerated only for a deliberate change to the numerics, and each such change is listed in the
:doc:`changelog <../whats_new>`.

Chunk-Invariance Tests
======================

* **Detect.** ``tests/detect/test_chunk_layout.py`` runs every anomaly method against both
  threshold methods, on the exact and the approximate path, across four time chunkings (whole, 17, 30
  and 60 steps) and three spatial layouts (including one with a deliberately small tile budget that
  forces several tiles). The outputs must be equal with no tolerance.
* **Tracker.** ``tests/track/test_merge_split.py`` varies the time chunking of the unstructured
  merge loop on synthetic cases. On an unstructured mesh the tracker is independent of chunking
  up to equidistant tie-breaks: when a cell is exactly as far from two candidate parents, the
  choice between them can differ between layouts . In a separate at-scale check, one real year of gridded 0.25° data
  gave identical tracked events across time chunks of 5 and 15 steps and across ``persist`` and
  ``streaming``.
* **Dimensions.** ``tests/detect/test_extra_dim.py`` includes a slice-equivalence check: a run on a
  three-dimensional field must equal the runs on each level separately.

The performance tests also assert chunk layouts, because an expensive all-to-all re-chunk can still
give identical values.

.. _validation-compute-modes:

Compute-Mode Equivalence
========================

``tests/detect/test_compute_mode_equivalence.py`` compares ``lazy`` and ``streaming`` with ``persist`` for five
method combinations (ten comparisons) plus two on an unstructured mesh, with no tolerance.
``tests/track/test_track_compute_mode.py`` does the equivalent for the tracker and also counts the bytes
each mode pins.

At full scale, on the global 0.25° dataset (3438 output days × 720 × 1440), ``streaming`` and
``persist`` gave identical values in every data variable . Only the coordinate encodings on disk differ between the modes.

Software Environments
=====================

Both major versions of Zarr are supported, and the suite is run in one environment for each:

.. list-table::
   :header-rows: 1

   * - Environment
     - Python
     - Zarr
     - xarray
   * - 1
     - 3.10
     - 2.18.3
     - 2025.6.1
   * - 2
     - 3.13
     - 3.4.0
     - 2026.9.0

Environment 2 also used dask and distributed 2026.8.0, numpy 2.5.3 and numcodecs 0.17.0.
``shifting_baseline`` with ``seasonal_percentile`` and the fixed-baseline methods gave
bit-identical arrays in the two environments. ``detrend_harmonic`` with ``standardise=True`` differs by
at most 9.5e-7 in ``STD``, ``dat_stn`` and ``thresholds_stn``. Rolling means run on the numpy path
(with ``bottleneck`` and ``numbagg`` switched off), so they are independent of the chunk layout and
the xarray version.

The continuous-integration workflow is configured for selected test files on Linux, macOS and
Windows (Python 3.10 on Linux only, 3.11 to 3.13 on all three), and for a coverage run over
``tests/`` on Linux and Python 3.11. ``tests/test_api.py`` fixes the public surface, so a removed
name fails a test.

Percentile Accuracy
-------------------

``tests/extremes/test_percentile_accuracy.py`` pins the relationship between the approximate and
exact percentile paths. The seasonal approximate threshold returns, to within 1.5 histogram bins, the
sample of rank ``floor(q * n) + 1``, which for the percentiles tested is NumPy's ``higher`` (upper
tail) or ``lower`` (lower tail) interpolation. The exact path uses linear interpolation, so the two
can differ by tens of bins on the sparse per-day-of-year samples at extreme percentiles (within about
8 bins on the global path). Where fewer than 10 samples lie beyond the threshold, marEx logs a
warning.

.. _validation-development-note:

Development Note
================

The scientific methods in marEx, and the implementation up to and including v4.1 in April 2026,
were developed by hand. From v4.1 onward I have used Claude Code, an AI coding assistant, to help me
optimise, generalise and test the package. That covers the work on three-dimensional fields,
non-daily cadences, the lower tail, the larger-than-memory modes and the test suite described above.

I do not take the output of that work on trust. The tracker's reference-output tests date from June
2026 and the detection tests from July 2026, and since then every change has been held to them at
zero tolerance, together with the chunking and compute-mode invariance tests, and the claims about
scale come from measurements on real data and not from reading the code. Where a result changed
on purpose, because an earlier behaviour was wrong or depended on the chunk layout, the changelog says
so and the reference outputs were regenerated with the difference recorded. This page is meant to
let you judge the package on that evidence. See :doc:`../whats_new` and the ``CHANGELOG.md`` at the
repository root for the corrections.
