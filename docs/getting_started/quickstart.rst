==========
Quickstart
==========

marEx runs in three stages: anomalies, extremes, tracking. Each can be used alone. The examples below build a small synthetic daily field (12 years on a 5-degree global grid, with a few warm blobs added) so everything runs in seconds. For real data see the :doc:`../tutorials/index`.

Synthetic Data
==============

.. code-block:: python

   import numpy as np
   import pandas as pd
   import xarray as xr
   from scipy.ndimage import gaussian_filter

   import marEx

   rng = np.random.default_rng(0)
   time = pd.date_range("2000-01-01", periods=12 * 366, freq="D")
   time = time[time.dayofyear <= 365][: 12 * 365]   # drop day 366: too few samples for a threshold
   lat = np.arange(-87.5, 90, 5.0)
   lon = np.arange(2.5, 360, 5.0)
   nt, ny, nx = len(time), len(lat), len(lon)

   season = 2.0 * np.sin(2 * np.pi * (time.dayofyear.values - 80) / 365.25)
   data = gaussian_filter(rng.standard_normal((nt, ny, nx)), sigma=(3, 2, 2))
   data += season[:, None, None] + np.linspace(0, 0.5, nt)[:, None, None]

   # Add three short-lived warm blobs, each drifting one cell a day
   yy, xx = np.meshgrid(np.arange(ny), np.arange(nx), indexing="ij")
   for t0, y0, x0 in [(2000, 10, 20), (3100, 25, 50), (3900, 15, 35)]:
       for k in range(3):
           r2 = (yy - y0) ** 2 + (xx - (x0 + k)) ** 2
           data[t0 + k] += 4.0 * np.exp(-r2 / 20.0)

   sst = xr.DataArray(
       data, coords={"time": time, "lat": lat, "lon": lon}, dims=("time", "lat", "lon"), name="sst"
   ).chunk({"time": 100})

Step 1: Anomalies Only
======================

``marEx.anomaly.compute`` removes the seasonal cycle and trend, with no thresholding. Use it when you want a climatology or anomaly field and nothing else.

.. code-block:: python

   anom = marEx.anomaly.compute(sst, method="shifting_baseline", window_years=5)
   anom.dat_anomaly      # (time, lat, lon), float32
   anom.mask             # (lat, lon), True where the first timestep is finite

``shifting_baseline`` takes the climatology from the previous ``window_years`` years, so the first ``window_years`` of the series are dropped from the output.

Step 2: Extremes
================

``marEx.preprocess_data`` chains the anomaly stage and the threshold stage. Here the 95th percentile of each day of year, pooled over an 11-day window, flags the extremes.

.. code-block:: python

   ds = marEx.preprocess_data(
       sst,
       method_anomaly="shifting_baseline",
       method_extreme="seasonal_percentile",
       threshold_percentile=95,
       window_years=5,
   )
   ds.extreme_events     # (time, lat, lon), bool
   ds.thresholds         # per-cell threshold for each day of year

For cold spells or droughts, add ``tail="lower"`` (with ``threshold_percentile=5`` for the coldest 5 %). If you already have anomalies from elsewhere, ``marEx.extremes.identify`` runs the threshold stage alone.

Step 3: Tracking and a Plot
===========================

The tracker needs a ``dask.distributed`` client, a boolean event field, and ``R_fill``, the radius in grid cells of the morphological closing.

.. code-block:: python

   # In a script, run everything from here under `if __name__ == "__main__":`,
   # because the local cluster starts worker processes that re-import the script.
   client = marEx.helper.start_local_cluster(n_workers=2, threads_per_worker=2, memory_limit="3GB")

   events = marEx.tracker(
       ds.extreme_events,
       ds.mask,
       R_fill=2,
       area_filter_absolute=5,
   ).run()

   events.ID_field       # (time, lat, lon), int32; 0 is background
   events.area           # (time, ID)
   events.time_start     # (ID)

   # The three planted blobs are the largest events (start dates 2005-06-25, 2008-06-29, 2010-09-08)
   peak_area = events.area.max("time")
   for ID, cells in peak_area.to_series().nlargest(3).items():
       print(ID, str(events.time_start.sel(ID=ID).values)[:10], cells)

   fig, ax, im = (events.ID_field > 0).mean("time").plotX.single_plot(
       marEx.PlotConfig(var_units="Event frequency", cmap="hot_r", cperc=[0, 96])
   )

The tracker keeps the spatial dimensions whole and chunks in time. Pass ``return_merges=True`` to ``run`` to also get the merge and split ledger. Datasets too large for memory are handled with ``compute_mode="streaming"``, covered in the :doc:`../guide/performance` guide.

Next Steps
==========

* :doc:`../guide/index` for how each stage works and how to choose methods.
* :doc:`../tutorials/index` for full notebooks on gridded, regional and unstructured data.
* :doc:`../whats_new` if you are upgrading from 4.x.
