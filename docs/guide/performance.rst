.. _performance-guide:

==================================
Performance and Larger-than-Memory
==================================

Two quantities decide whether a marEx run finishes, and they respond to different settings.

The **per-task working set** is the memory one task needs while it runs. It is fixed by the
algorithm and the chunk shape, and it is the same in every ``compute_mode``. If a single task is
larger than one worker's ``memory_limit``, the run dies however many workers are added.

**Pinned bytes** are the arrays a stage holds in cluster memory between tasks so that later steps
can reuse them. ``compute_mode`` controls these, and nothing else. This page covers both, then
the cluster setup that makes the limits real, and ends with the runs that were measured.

.. contents::
   :local:
   :depth: 2

.. _performance-compute-mode:

Choosing a Compute Mode
=======================

``compute_mode`` is accepted by :func:`marEx.preprocess_data`, :func:`marEx.anomaly.compute` and
:class:`marEx.tracker`. It decides what happens to the intermediate arrays that more than one later
step reads, such as the anomaly or the labelled event field.

.. list-table::
   :header-rows: 1
   :widths: 14 38 48

   * - Mode
     - What happens to those intermediates
     - Use it when
   * - ``'persist'`` (default)
     - Computed once and pinned in cluster memory.
     - The pinned total fits in aggregate worker memory. This is the simplest mode and the one
       with the most coverage.
   * - ``'lazy'``
     - Nothing is pinned and upstream work is recomputed wherever it is read again.
       Detect only. The tracker rejects it with ``ConfigurationError``, because its merge loop
       is sequential in time and recomputation buys nothing.
     - Memory is tight and recomputation is acceptable. Inside ``preprocess_data`` the input is
       read three times, plus once more for every traversal of the outputs, so
       materialise what you need in one pass.
   * - ``'streaming'``
     - Written to Zarr in a staging directory and re-opened, so the arrays live on disk instead
       of in memory. Needs ``scratch_dir`` (detect) or ``temp_dir`` (tracker).
     - The pinned total does not fit, or you want the cluster to stay small as the record gets
       longer.

How the modes differ per stage:

* **Anomaly and extremes (detect).** ``'persist'`` pins the float32 anomaly and the event flags,
  about ``n_time × n_cells × 5`` bytes, plus the thresholds. ``'streaming'`` moves them to disk.
  It does not lower the per-task working set, so the peak transient of a detect run is largely
  unchanged by the mode (see :ref:`performance-measured`).
* **Tracker.** ``'streaming'`` stages the whole-field intermediates, the filled and filtered
  fields, and the output accumulator, so memory scales with the time chunk and not with the length of
  the series. The pinned bytes of the gridded tracker at 3804 steps fell from 337 GB to
  0.34 GB. This is where streaming changes whether a run completes. It supports both gridded and
  unstructured input.
* **Peak versus pinned.** Streaming cuts *pinned* bytes. The peak memory of the process falls
  by much less, because peak also contains terms that do not depend on the mode. Report the pinned
  bytes, and treat any peak as a property of the room the run was given: the same workload peaked at
  22 GB in a 32 GB budget and at 57 GB in a 96 GB budget, because dask expands into the memory it
  has and releases it under pressure.

``'lazy'`` has been checked against ``'persist'`` for correctness on small fields only. Its
memory and runtime behaviour at scale have not been measured, so no figure is given for it here.

Equivalence between the modes is part of the test suite: see :ref:`validation-compute-modes`.

.. _performance-staging:

Streaming Staging and ``clear_staging``
---------------------------------------

A ``'streaming'`` result is lazy. It reads from the staging directory, so that directory has to
outlive the call that created it. Write the output first, then clear the staging.

.. code-block:: python

   import marEx

   extremes = marEx.preprocess_data(
       sst,                             # dask-backed DataArray
       method_anomaly="shifting_baseline",
       method_extreme="seasonal_percentile",
       threshold_percentile=95,
       compute_mode="streaming",
       scratch_dir="/path/to/fast/scratch",
   )

   extremes.to_zarr("extremes.zarr", mode="w")   # write first
   marEx.clear_staging(extremes)                 # then remove the staging directory

The path is stored in ``extremes.encoding["marex_staging_dir"]``. It is deliberately not in
``attrs``, because attributes are copied into whatever you write the dataset to and the directory is
gone by the time anyone reads that file back. The same contract holds for the tracker:
``events_ds.encoding["marex_staging_dir"]``.

Three practical points.

* ``clear_staging`` accepts either the dataset or the path, and is safe to call twice.
  ``xr.merge`` and ``xr.concat`` drop dataset-level encoding, so after either of them
  ``clear_staging(ds)`` can only warn. Keep the path from before the merge.
* The cleanup that runs at interpreter exit does not survive ``SIGKILL``. A run killed by the
  scheduler's wall clock can leave its ``marex_stage_*`` directory behind, so sweep the scratch
  directory from time to time.
* Budget disk before a long tracker run: staging writes about 14 bytes per cell-timestep
  uncompressed, which is about 55 GB for 3804 steps on a 720 × 1440 grid and about 230 GB for 1096
  steps on a 14.9 M-cell mesh. Compression makes the real footprint smaller.

Streaming also changes the chunk layout of the returned ``dat_anomaly`` (staging rewrites it to
uniform chunks). The values are identical.

The tracker in streaming mode needs uniformly chunked time. A ragged chunking is re-chunked with
a warning, and a ``temp_dir`` is required. See :doc:`tracking` for the tracker side.

.. _performance-chunking:

Chunking
========

The three stages want different layouts, and the layout also depends on the grid. Re-chunking
between stages is correct, not something to optimise away.

.. list-table::
   :header-rows: 1
   :widths: 24 38 38

   * - Stage and grid
     - Space
     - Time
   * - Detect, lat/lon grid
     - Whole is correct. The anomaly and threshold reductions are global in time and independent
       in space, and the output is always spatially whole.
     - Modest, at least ``smooth_days``. The default ``dask_chunks={"time": 25}`` is a sound
       start.
   * - Detect, large unstructured mesh
     - **Chunk the cell dimension**, below about 200 million / ``n_time`` cells per chunk (see below).
       The shipped example uses 15,625.
     - At least ``smooth_days``. The shipped example uses 365 steps (23 MB chunks).
   * - Tracker, either grid
     - **Whole** (``-1``). Connected-component labelling and the dilation matrix are global in space.
     - The only knob. Small chunks: 25 on the lat/lon example, 5 on the unstructured one.

Detect on a Large Mesh
----------------------

The percentile reductions regroup each spatial tile's full time series before they build the
per-cell histograms. On a large mesh, one task therefore holds approximately

.. code-block:: text

   n_time × cells in one INPUT spatial chunk × 4 bytes

and this depends on the spatial width of your input chunks. The time chunk of the input does not
reduce it.

On the ICON R02B09 mesh (14.9 M cells, 8 years, 2922 steps), input chunks of
``{"time": 21, "ncells": -1}`` imply a 174 GB tile. That configuration made no progress in
5 h 40 min, because the internal re-chunk becomes an all-to-all transpose. With
``{"time": 21, "ncells": 25000}`` the tile is 292 MB. The failure is a cliff and not a slope: the
workers pause and resume, memory is reported as unmanaged and high, and nothing progresses.

The anomaly stage keeps your spatial chunk as its internal tile while one task on it, ``n_time`` × cells per
chunk, stays within about 200 million elements (four times the 50-million-element task budget).
Above that it re-tiles the input into smaller tiles that straddle your chunks, and the task graph
grows several-fold. In the 30-year ICON example (10,957 steps) the limit is about 18,000 cells, and
the example uses 15,625.

Leaving space whole is therefore wrong on a large unstructured mesh. The advice does not transfer to a
lat/lon grid: the global 0.25° runs in :ref:`performance-measured` used spatially whole input chunks,
``{"time": 25, "lat": -1, "lon": -1}``, and completed.

Match the Store
---------------

Choose a chunking that can be reached from the on-disk chunks by local merges and splits. A jump
from a time-chunked store straight to ``{"time": -1, ...}`` makes every output chunk depend on
every input chunk. Check what you have before choosing:

.. code-block:: python

   import xarray as xr

   ds = xr.open_zarr("sst.zarr", chunks={})
   print(dict(ds.sst.sizes), ds.sst.chunks)

Tracker Chunking
----------------

Because space stays whole, one chunk is ``time_chunk × n_cells``. On the ICON mesh a boolean chunk of
4 steps is 59.5 MB, and on the 720 × 1440 grid a chunk of 25 steps is 25.9 MB. Interior time chunks
should be uniform; a shorter final chunk, including a single step, is accepted.

Output Chunking
---------------

``dask_chunks`` on :func:`marEx.preprocess_data` sets the chunking of the returned dataset, and only its
time entry is honoured. Horizontal dimensions are always whole, because the tracker needs them
that way. ``"auto"`` means dask's byte budget (``array.chunk-size``), not an element count. An
integer is a number of steps, and extra dimensions such as depth are sized under about
50 million elements per chunk where one level allows it. The cycle axis (``dayofyear``,
``month`` or ``hourofyear``) is chunked from the same entry, capped at the cycle length.

Chunk the output for the next stage and not the current one. Zarr requires uniform chunks (the
last may be shorter), so a ragged layout left behind by an intermediate step has to be regularised
before ``to_zarr``.

.. _performance-tile-warning:

The Tile-Fit Warning
--------------------

The internal re-chunk holds time whole and caps the spatial tile near 50 million elements per task,
where each cell counts the larger of its number of timesteps and the cycle length. The bin count
does not enter this budget. A ``window_spatial`` floor overrides the cap, because a rolling window
may not cross a chunk boundary, so the warning fires only when ``n_time × window_spatial²`` alone
exceeds the budget. At the gridded default of ``window_spatial=5`` that takes more than 2 million
timesteps, over two centuries of hourly data. marEx logs a warning that names the estimate, the
budget and the window that forced it. The warning never changes the tile and therefore cannot
change a result. The levers are a narrower ``window_spatial``, a shorter record, fewer threads per
worker, or ``compute_mode="streaming"``. See also :ref:`troubleshooting-tile-warning`.

.. _detect-memory-sizing:

Memory Sizing
=============

Two terms set what a detect run needs.

**Pinned output** (``'persist'``) is about ``n_time × n_cells × 5`` bytes for the float32 anomaly
and the boolean event flags, plus ``n_cycle × n_cells × 4`` bytes for the thresholds. For global
0.25° data (1,036,800 cells) over 3438 output days with day-of-year thresholds that is
14.3 + 3.6 + 1.5, or about 19 GB across the cluster, which is 1.2 GB per worker on 16 workers.
Under ``'streaming'`` the anomaly and thresholds end up on disk and the event flags are computed as
the output is written, so most of this term drops away. Arrays that scale with space only
(thresholds, the cumulative histogram, the daily climatology) do not shrink with the time chunk
in any mode.

**Histogram working memory** (every mode). The ``approximate`` percentile path regroups each
spatial tile's full time series before it builds the per-cell histograms. Each task is bounded, but
the shuffle buffers and the tasks running together on a worker are not. Every worker restart in the
measured runs below fell in this stage, in both modes, so ``'streaming'`` does not remove it. Under
``'lazy'`` the anomaly is recomputed inside the same stage, which adds to it.

The measured runs used global 0.25° daily data (7091 input days), ``shifting_baseline`` with
``seasonal_percentile``, input chunks ``{"time": 25, "lat": -1, "lon": -1}`` and 16 workers of 4
threads and 14 GB each:

.. list-table::
   :header-rows: 1

   * - Mode
     - Outcome
   * - ``'persist'``
     - Completed with 6 worker restarts (two runs; an earlier development version of the histogram
       stage failed here with ``KilledWorker`` after 21 restarts).
   * - ``'streaming'``
     - Completed with 2 worker restarts (one run).

These runs used an earlier version of the histogram stage. A later per-cell kernel, before the
data-derived default bins, completed the two global 40-year rows in the table below at no more than
about 1.4 GB per thread (cgroup peak, page cache included), so the figures here are a conservative
bound. For that earlier stage at this size, treat 14 GB per 4-thread worker as the edge, and
budget above it. Size ``memory_limit`` from the largest single worker's peak, which
:class:`~marEx.helper.ResourceMonitor` reports per stage, not from the summed peak. In the detect
runs of the example notebooks the largest worker reached 5.3 GB of 7.8 GB on single-threaded workers
(global 0.25°, 40 years of input, ``persist``) and 24.4 GB of 31.2 GB on 8-thread workers (ICON,
30 years of input, ``streaming``). Each is one run, sampled once a second, so the peaks are lower
bounds, and one run per shape does not give a crash rate.

A run at the edge logs the dask line ``exceeded 95% memory budget. Restarting...`` during extreme
identification (see :ref:`troubleshooting-worker-warnings` for why you may not see it). A restart
costs time, because dask recomputes the lost tasks. Two settings should move a run away from the
edge, though neither has been measured at full scale:

* **Fewer threads per worker** at the same ``memory_limit``, so fewer tiles are in memory at once.
* **A larger ``memory_limit``** per worker with the same thread count.

Reduce threads before reducing workers. Threads multiply the number of concurrent tasks inside one
limit, workers do not. Aggregate memory never rescues a task that is larger than one worker's limit.

For ``method_percentile="exact"`` the full time series of a cell must be resident, and the tiling is
bounded by the same 50-million-element task budget. The exact path completed a 20-year global 0.25°
run with 16 workers of 6 GB and 4 threads (see :ref:`performance-measured`).

The tracker has its own sizing. In ``'persist'`` the dominant pin is the whole int32 event field,
``n_time × n_cells × 4`` bytes, and the client process also grows with it (the dask
``memory_limit`` bounds the workers, not the client). Size a tracker squeeze from that
arithmetic and not from a measured peak.

.. _performance-clusters:

Clusters
========

Start the cluster before the first marEx call. The tracker needs a ``dask.distributed`` client and
the detect stages should use one, because without it there is no memory limit to bind.

.. code-block:: python

   import marEx

   # In a script, run this under `if __name__ == "__main__":` (its worker processes re-import the script).
   client = marEx.helper.start_local_cluster(
       n_workers=4,
       threads_per_worker=2,
       scratch_dir="/path/to/fast/scratch",
       memory_limit="12GB",        # per worker; always pass it
   )

``start_local_cluster(n_workers=4, threads_per_worker=1, scratch_dir=None, verbose=None,
quiet=None, **kwargs)`` applies :func:`marEx.helper.configure_dask`, builds a ``LocalCluster``
and passes any extra keyword arguments to it. It reduces the worker count if the requested threads
exceed the logical cores.

.. _performance-memory-limit:

Pass ``memory_limit`` Explicitly
--------------------------------

Dask reads the total memory of the machine it runs on. Under a batch scheduler, in a container or
under a cgroup limit it may not see the cap your job actually has. A ``LocalCluster`` then sizes
every worker as if it owned the node, no worker ever reaches its limit, and an under-provisioned run
behaves normally until the operating system kills the whole job. ``start_local_cluster`` sets no
limit for you. Pass ``memory_limit`` per worker, and write the effective value to your log:

.. code-block:: python

   workers = client.scheduler_info()["workers"].values()
   limits_gb = sorted({w["memory_limit"] / 1e9 for w in workers})
   print(f"{len(workers)} workers, per-worker memory_limit (GB): {limits_gb}")

   # n_workers x memory_limit, plus room for the client and scheduler, must fit the allocation
   assert len(workers) * limits_gb[0] < 0.9 * 96, "workers exceed the job's memory"

The scheduler and the client live in the same allocation as the workers. A large problem can need
10 GB or more in the client alone, and that cost is the same in every ``compute_mode``. If a job is
killed, compare the accounting record (for example SLURM's ``sacct MaxRSS`` against ``ReqMem``)
before blaming the setting you were testing. :class:`~marEx.helper.ResourceMonitor` (below) makes
this comparison for you at start-up.

Do Not Disable Spilling
-----------------------

Leave dask's spill-to-disk on. Switching it off does not sharpen a memory test. A worker that
crosses the ``pause`` threshold can no longer spill back down and pauses permanently, so the cluster
deadlocks, the peak rises because what would have spilled stays resident, and both compute modes fail.
:func:`marEx.helper.configure_dask` applies marEx's defaults: worker memory ``target``, ``spill``,
``pause`` and ``terminate`` at dask's own 0.6, 0.7, 0.8 and 0.95, work stealing off, an ``array.chunk-size`` of
24 MiB and 300 s communication timeouts. It sets ``temporary_directory`` to a fresh directory under
``scratch_dir`` and returns the object that owns it, which must stay alive. Pass ``scratch_dir``
explicitly: the default is a site-specific path.

SLURM
-----

On a SLURM system with ``dask-jobqueue`` installed, :func:`marEx.helper.start_distributed_cluster`
submits workers as batch jobs.

.. code-block:: python

   import marEx

   client = marEx.helper.start_distributed_cluster(
       n_workers=16,
       workers_per_node=4,
       runtime=120,           # minutes
       node_memory=256,       # GB per node: 256, 512 or 1024
       queue="compute",
       account="your_project",
       scratch_dir="/path/to/fast/scratch",
   )

The signature is ``start_distributed_cluster(n_workers, workers_per_node, runtime=9,
threads_per_worker=1, node_memory=256, dashboard_address=8889, queue='compute', scratch_dir=None,
account=None, verbose=None, quiet=None, **kwargs)``. Each node runs ``workers_per_node`` processes of
``threads_per_worker`` threads. Fewer, multi-threaded workers mean fewer peers in every all-to-all
transfer. Without ``account`` the worker jobs are charged to ``$SLURM_ACCOUNT``, ``$SBATCH_ACCOUNT`` or
``$SLURM_JOB_ACCOUNT``, the first that is set, or else to the account of the SLURM job the call runs
in. If none of these gives an account it raises ``ConfigurationError``. The function was written
for the DKRZ Levante machine: the defaults for network interface and log directory are specific
to it, and ``node_memory`` outside 256, 512 and 1024 raises ``ConfigurationError``. On another site, construct a ``dask_jobqueue`` cluster directly with the
same memory and spill settings, or run marEx inside a single batch allocation with
``start_local_cluster`` and an explicit ``memory_limit``. Inside an allocation, keep
``n_workers × memory_limit`` below the memory you were granted.

``marEx.helper.get_cluster_info(client)`` returns the host name, scheduler port and dashboard link,
and prints the port-forwarding hint for reaching the dashboard from your laptop.

.. _performance-resource-monitor:

Measuring a Run
---------------

:class:`marEx.helper.ResourceMonitor` records, stage by stage, the wall time, the peak summed worker
memory, the peak of the largest single worker, the peak client memory, the peak bytes spilled to disk
and the number of worker restarts.

.. code-block:: python

   import marEx

   # In a script, run this under `if __name__ == "__main__":` (its worker processes re-import the script).
   client = marEx.helper.start_local_cluster(n_workers=4, memory_limit="12GB")
   monitor = marEx.helper.ResourceMonitor(client, interval=1.0)

   with monitor.stage("detect"):
       extremes = marEx.preprocess_data(sst, compute_mode="streaming", scratch_dir="/path/to/fast/scratch")
       extremes.to_zarr("extremes.zarr", mode="w")
       marEx.clear_staging(extremes)

   print(monitor.summary())   # one row per stage, plus a total row

A background thread samples the scheduler every ``interval`` seconds, so a spike shorter than that
can be missed and the peaks are lower bounds. On construction the monitor logs each worker's
``memory_limit`` and warns if the limits of the workers on this machine add up to more than the
memory of the job (the cgroup limit when one is set, otherwise the SLURM grant). Workers started as
their own SLURM jobs on other nodes do not count against it.

.. _performance-measured:

Measured at Scale
=================

These are single runs on the DKRZ Levante supercomputer (whole compute nodes unless stated), reported as
measured. They show what completed in one configuration. They are not guarantees, not benchmarks
of other machines, and none has been repeated enough to give a spread, except where a count is
given. Memory figures for the detect rows are the sum of worker RSS or the job's cgroup peak, as
labelled.

Detect
------

.. list-table::
   :header-rows: 1
   :widths: 30 22 16 16 16

   * - Dataset and method
     - Cluster
     - Wall
     - Memory
     - Runs
   * - 40 yr daily global 0.25° SST (9282 output days × 720 × 1440), ``shifting_baseline`` +
       ``seasonal_percentile``, approximate
     - 4 workers × 16 threads × 22 GB (one node, 64 threads)
     - ``persist`` 3954.5 s, ``streaming`` 3663.1 s
     - cgroup peak 86.6 and 85.3 GB, 0 restarts
     - 1 per mode
   * - Same data and method
     - 6 workers × 10 threads × 12 GB
     - ``persist`` 2576.7 s, ``streaming`` 2356.4 s
     - worker RSS peak 57.4 and 51.7 GB, 0 restarts
     - 1 per mode
   * - 19.4 yr global 0.25° (7091 input days, 3438 output days), same method
     - 16 workers × 4 threads × 14 GB
     - ``persist`` 8848.9 s, ``streaming`` 6520.6 s
     - ``persist`` 6 restarts, ``streaming`` 2 restarts
     - ``streaming`` 1, ``persist`` 2
   * - 40 yr regional OSTIA SST, 0.05° (14761 days × 800 × 1300), same method
     - 16 workers × 4 threads × 14 GB
     - 9759.8 s
     - 3 worker restarts in the histogram stage
     - 1
   * - 20 yr global 0.25°, exact percentiles
     - 16 workers × 4 threads × 6 GB
     - 2087.1 s
     - cgroup peak 79.4 GB of 116 GB, 0 restarts
     - 1
   * - ICON R02B09 (14.9 M cells), 12 yr (4383 days), run as 6 cell slabs
     - 30 GB per worker
     - 4563 to 5199 s per slab
     - spill peak 143 to 167 GiB per slab, 0 restarts
     - 1 per slab
   * - 20 yr 6-hourly global 0.25° SST (121.3 GB float32), 5 yr ``shifting_baseline`` +
       ``seasonal_percentile``, in a 32 GiB job
     - 3 workers × 2 threads × 9 GB
     - ``persist`` 11490.3 s, ``streaming`` 9984.5 s
     - cgroup anon peak 21.5 and 20.6 GiB, spill 161 and 138 GB, 0 OOM kills
     - 1 per mode
   * - ICON R02B09 (14.9 M cells), 30 yr input (7305 output days), ``streaming``, all cells in one run
     - 48 workers × 8 threads × 31.2 GB (6 nodes)
     - 36.6 min
     - largest worker 24.4 GB, spill peak 1061 GB, 0 restarts
     - 1

In the first and third rows ``persist`` and ``streaming`` produced identical data variables (the 19.4 yr
pair was compared file by file, 292 of 292). In the second row the thresholds, masks and events were
compared against the first row and agreed. In each pair ``streaming`` was no slower than
``persist``. The first two rows are the same data and method on two cluster shapes, so wall time
depends on the cluster as much as on the data. The regional run is marginal: three histogram
restarts, and it completed. The slab runs used the native (non-streaming) path, and three slabs ran
at once without slowing down. The whole-mesh ICON row is the shipped unstructured example notebook, peaks sampled once a second
by :class:`~marEx.helper.ResourceMonitor`.

The two ICON rows are the only unstructured detect measurements. Detect on a spatially whole input
of a large unstructured mesh is not advertised (see :ref:`performance-chunking`).

Tracking
--------

.. list-table::
   :header-rows: 1
   :widths: 30 22 16 16 16

   * - Dataset and mode
     - Cluster
     - Wall
     - Memory
     - Runs
   * - Global tracker from the 19.4 yr detect output (3438 days, 3116 events)
     - 16 workers × 14 GB class
     - ``persist`` 1140.6 s, ``streaming`` 629.2 s
     - not recorded. Outputs identical (event count, edges, ID field)
     - 1 per mode
   * - Gridded tracker, 0.25°, 3804 days, 24 GiB job allocation, 4 × 4 GB dask budget
     - 4 workers × 4 GB
     - ``streaming`` 1600 to 1676 s
     - ``persist`` OOM-killed. ``streaming`` completed with a peak of 6.65 to 7.30 GB
     - ``persist`` 5 of 5 killed, ``streaming`` 7 of 7 completed
   * - Same tracker, 96 GB budget
     - 96 GB budget
     - ``persist`` 1103.8 s, ``streaming`` 1099.5 s
     - peak 56.7 and 19.1 GB, pinned 337.0 and 0.340 GB
     - 1 per mode
   * - ICON R02B09 tracker, 1096 days, 4359 events
     - 16 workers × 12 GB
     - ``persist`` 4297 and 4302 s, ``streaming`` 4219 s
     - peak 126.9 and 128.2 GB (``persist``), 118.7 GB (``streaming``). Pinned 751.4 GB falls
       to 147.5 GB
     - ``persist`` 2, ``streaming`` 1
   * - Gridded tracker, 0.25°, 15 yr of 6-hourly extremes (91.0 GB int32 ID field), 32 GiB job
     - 3 workers × 2 threads × 9 GB
     - ``streaming`` 8316.5 s
     - ``streaming`` anon peak 9.4 GiB. ``persist`` OOM-killed after 86 min with the workers under
       their limits and the client at 14.6 GB
     - 1 per mode

What these rows say:

* The headline for larger-than-memory work is the second row. In a 24 GiB allocation ``persist``
  was OOM-killed on all five attempts while ``streaming`` completed on all seven, and the streaming
  peak stayed near 7 GB while the field grew from 3.9 GB to 15.8 GB over series lengths of 951, 1902
  and 3804 days. Read that as "near-flat in series length". The seven runs at 3804 days spread over
  0.65 GB, which is the size of the growth being measured, so nothing sharper than that is
  defensible. At 28 GiB ``persist`` did not finish within its deadline and at 40 GiB it completed,
  so the result is about this allocation and this budget, not a general threshold.
* The 6-hourly rows keep one 32 GiB job fixed while the record grows to 3.5 times its memory. The
  ``streaming`` peak moved 6 % for detect when the input doubled, and under 1 % for track when the
  ID field tripled. ``persist`` detect also completed, by spilling more, so detection at this size
  does not need ``streaming``. ``persist`` track did not complete. Before the job hit its limit the
  client, which runs the merge loop, had grown to 14.6 GB while the workers stayed under theirs.
* At a 96 GB budget the wall times of the two runs, one per mode, differ by under 1 %. Streaming is
  not a speed-up. Its effect is on pinned bytes, 337.0 GB to 0.340 GB, and the peak fell as well because
  this is the one wide-window case.
* On ICON ``persist`` completes the 1096-step track at 192 GB total. Streaming cut pinned bytes
  5.1 times but the peak by only 6.5 %. At 4 × 8 GB (32 GB), ``persist`` was stopped by a 5 h
  harness deadline on both attempts, with no out-of-memory event recorded, and ``streaming``
  completed in 3 h 50 min. That is a wall-clock statement and not a statement that ``persist``
  cannot fit.

The two 6-hourly rows are reproduced by the notebook and batch job in
`examples/larger-than-memory <https://github.com/wienkers/marEx/tree/main/examples/larger-than-memory>`_, which ship each run's summary. The other rows were
measured with an earlier harness that is no longer distributed.

Choosing a Strategy
===================

* If the data fit in the aggregate memory of the cluster you can get, scale out and stay in
  ``'persist'``. Add workers before reaching for streaming.
* If the *tracker* needs a field larger than the cluster can pin, use ``'streaming'`` and give it
  ``temp_dir``. This is where the measured benefit is.
* For *detect* on a large mesh, fix the spatial chunking first, since a mis-chunked run fails in
  every mode. Then size workers against the per-task working set, then pick the mode.
* A failed run is a symptom of a memory problem more often than a software one. Go through
  :doc:`../troubleshooting` before changing the mode.
