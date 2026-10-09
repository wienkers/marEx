.. _validation-guide:

==============================
Validation and Reproducibility
==============================

marEx is checked in three ways: its outputs are compared with stored reference outputs at zero
tolerance, the same answer is required from different chunkings and compute modes, and the claims
that need real scale are measured on real data. This page says what each check covers and, as
important, what it does not.

The test suite lives in ``tests/`` and holds about 1,400 tests. The most recent full run in both
environments listed below passed 1390 tests with 4 skipped and no failures.

Reference-Output Tests
======================

``tests/test_pipeline_golden.py`` pins :func:`marEx.preprocess_data` on two configurations of a
deterministic gridded fixture, using the last seven years (2555 daily steps):

* ``detrend_harmonic`` with ``global_percentile``, which exercises the one-dimensional histogram path.
* ``shifting_baseline`` with ``seasonal_percentile``, which exercises the per-day-of-year path.

Both use approximate percentiles. The stored reference outputs are Zarr stores in ``tests/data/``.
Every comparison in these reference tests is made at zero tolerance. The one exception is the thresholds of
the first configuration, which are compared with an absolute tolerance of ``2e-14``. That is float64
round-off (the largest difference seen was 1.1e-14, on 732 of 800 cells), and it is small enough that
it could not hide a change in the algorithm. Integer and label outputs have no tolerance at all.

The reference runs fix ``precision=0.01`` and a pinned range, so they characterise the numerics
and do not move when the default bin choice changes. ``test_legacy_bins_reproduce_the_goldens`` isolates
the histogram path: it reproduces the references with the legacy bin layout, so a failure there
points at the histogram code and not at the surrounding pipeline. ``tests/test_track_golden.py``
does the same for the tracker, against a stored event dataset and merge dataset.

Reference outputs have been regenerated when a change altered numerics on purpose, for example when
the threshold computation was made independent of chunking and when rolling means were moved onto a
single numerical path, and the differences were recorded at the time. Those deliberate changes
are listed in the :doc:`changelog <../whats_new>`, so a reference that moves is a documented event
and never a quiet one.

Chunk-Invariance Tests
======================

A result that depends on how the data happens to be chunked is hard to trust, so chunking is varied
on purpose.

* **Detect.** ``tests/test_detect_chunk_invariance.py`` runs every anomaly method against both
  threshold methods, on the exact and the approximate path, across four time chunkings (whole, 17, 30
  and 60 steps) and three spatial layouts (including one with a deliberately small tile budget that
  forces several tiles). The outputs must be equal with no tolerance. This is verified on the test
  fixtures, for every anomaly and threshold method.
* **Tracker.** ``tests/test_merge_chunk_invariance.py`` varies the time chunking of the unstructured
  merge loop on synthetic cases. On an unstructured mesh the tracker is independent of chunking
  except for equidistant tie-breaks: when a cell is exactly as far from two candidate parents, the
  choice between them can differ between layouts. In the one real record where this was measured it
  affected 91 cells on 2 days, at the boundary of one of 692 events. On a gridded record, one real
  year of 0.25° data gave identical tracked events across time chunks of 5 and 15 steps and across
  ``persist`` and ``streaming`` (a separate at-scale check, not part of the suite). Bit-identity is
  therefore not claimed for the unstructured tracker across chunkings.
* **Dimensions.** ``tests/test_3d_preprocessing.py`` includes a slice-equivalence check: a run on a
  three-dimensional field must equal the runs on each level separately.

These tests check structure as well as values. Identical values do not show that the task graph is
sensible, and an expensive all-to-all re-chunk can pass a value test, so the performance tests also
assert chunk layouts.

.. _validation-compute-modes:

Compute-Mode Equivalence
========================

``tests/test_compute_mode_equivalence.py`` compares ``lazy`` and ``streaming`` with ``persist`` for five
method combinations (ten comparisons) plus two on an unstructured mesh, with no tolerance.
``tests/test_track_compute_mode.py`` does the equivalent for the tracker and also counts the bytes
each mode actually pins, because checking whether an array is a dask collection says nothing about
whether it was persisted.

At full scale, on the global 0.25° dataset (3438 output days × 720 × 1440), ``streaming`` and
``persist`` were compared on every data variable and found identical: 0 differing elements of
3,564,518,400 for the largest array, and 0 of 379,468,800 and 1,036,800 for the others. This is one
configuration, run once. The coordinate encodings on disk differ between the modes, and the values
do not.

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
(with ``bottleneck`` and ``numbagg`` switched off) so that they do not depend on the chunk layout or
on the xarray version. Python 3.11 and 3.12 were not run in these two environments.

The continuous-integration workflow runs the suite on Linux, macOS and Windows for Python 3.10 to
3.13. ``tests/test_api_surface.py`` fixes the public surface, so a removed name or a threshold
parameter appearing on :func:`marEx.anomaly.compute` fails a test.

Percentile Accuracy
-------------------

The approximate and exact percentile paths use different conventions, and
``tests/test_percentile_agreement.py`` pins the relationship. The seasonal approximate threshold
matches NumPy's ``higher`` (upper tail) or ``lower`` (lower tail) interpolation to within 1.5
histogram bins. Against the exact path, which uses linear interpolation, the gap can reach
tens of bins on the sparse per-day-of-year samples at extreme percentiles (on the global path it
stays within about 8 bins). The cause is the thin sample behind each day-of-year window, not the bin width.
Where fewer than 50 samples lie beyond the threshold, marEx logs a warning.

What Is Not Claimed
===================

* **No full-scale bit-identity for the ICON mesh.** Identity between modes and layouts is shown at
  fixture scale on both grid types, and at global 0.25° scale on the lat/lon grid. It has not been
  shown for a complete ICON R02B09 run.
* **Most at-scale comparisons were run once.** Where a configuration was repeated, the count is
  given in :ref:`performance-measured`. There is no spread on the others.
* **Memory caps differ between runs.** Some at-scale runs had a SLURM memory cap on the job, others
  were bounded only by Dask's per-worker ``memory_limit``.
* **The performance of** ``lazy`` **is unmeasured.** Its correctness is tested on small fields.
* **The unstructured tracker excludes equidistant tie-breaks**, as above, and the seasonal slab path
  has not been run on an unstructured mesh.
* **Sub-daily data with** ``shifting_baseline`` **is not advertised.** ``detrend_harmonic`` rejects it.

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
