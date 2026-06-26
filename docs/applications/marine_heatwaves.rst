================
Marine Heatwaves
================


Marine heatwaves are the most developed use of marEx. They are prolonged, spatially
coherent periods of anomalously warm sea surface temperature, and their footprints matter
as much as their intensity. A heatwave over a reef system, a kelp forest or a set of
aquaculture leases is a statement about where the warm water was and for how long, which a
time series at one grid cell cannot give.

Why marEx Fits
==============

* The threshold is seasonal and local. ``seasonal_percentile`` computes one threshold per
  cell for each day of the year, pooling a window of days (``window_days``) and, on gridded
  data, neighbouring cells (``window_spatial``). This follows the form of the Hobday et al.
  (2016) definition, with a configurable percentile in place of a fixed 90th.
* The baseline can shift. With ``shifting_baseline`` the climatology is a rolling mean over
  the preceding ``window_years`` (15 by default), so the warming trend is not counted as
  extreme year after year. The first ``window_years`` years are trimmed from the output.
* Events are tracked as objects. The tracker joins exceeding cells into events with an area,
  a centroid track, a start and an end, and handles splits and merges, so a heatwave that
  breaks into two and rejoins is one record rather than two.
* The whole chain is lazy. Detection works on a Dask array, and ``compute_mode="streaming"``
  keeps the intermediate fields on disk when they would not fit in memory (see
  :doc:`../guide/performance`).

Configuration
=============

The snippet builds a small synthetic SST field (30 years, daily, 20 x 30 grid, a warm patch
planted for 40 days in 2015 and a land block of NaNs), then runs detection and tracking.
``window_years`` is left at its default of 15, so the output covers 2005 to 2019. Real use
differs in the input and the sizes: ``R_fill`` and the area filter are chosen for the grid
spacing.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import xarray as xr
   import marEx


   if __name__ == "__main__":

       # Synthetic daily SST: seasonal cycle, warming trend, noise and one warm blob (20 x 30 grid, 30 years)
       rng = np.random.default_rng(0)
       time = pd.date_range("1990-01-01", "2019-12-31", freq="D")
       lat = np.linspace(-60, 60, 20)
       lon = np.linspace(0, 348, 30)
       doy = time.dayofyear.values[:, None, None]
       sst = (
           15 + 8 * np.cos(2 * np.pi * (doy - 30) / 365.25) * np.cos(np.deg2rad(lat))[None, :, None]
           + 0.3 * np.arange(time.size)[:, None, None] / 365.25
           + 0.5 * rng.standard_normal((time.size, lat.size, lon.size))
       )
       # a 40-day warm event over a 6 x 8 patch, in 2015
       sst[9140:9180, 6:12, 10:18] += 3.0
       sst[:, 0:3, 0:4] = np.nan  # "land", NaN at every timestep
       da = xr.DataArray(sst.astype("float32"), dims=("time", "lat", "lon"),
                         coords={"time": time, "lat": lat, "lon": lon}, name="sst")
       da = da.chunk({"time": 100})

       client = marEx.helper.start_local_cluster(n_workers=2, threads_per_worker=2, memory_limit="3GB")

       ds = marEx.preprocess_data(
           da,
           method_anomaly="shifting_baseline",   # rolling 15-year climatology (the default window)
           method_extreme="seasonal_percentile",
           threshold_percentile=95,
           )

       tr = marEx.tracker(ds.extreme_events, ds.mask, R_fill=2, area_filter_absolute=20, grid_resolution=10.0)
       events = tr.run()
       dur = (events.time_end - events.time_start) / np.timedelta64(1, "D") + 1   # days
       longest = dur.idxmax("ID")                                                 # ID of the longest-lived event
       print("longest event:", int(longest), "|", float(dur.sel(ID=longest)), "days | starts",
             str(events.time_start.sel(ID=longest).values)[:10], "| peak area",
             float(events.area.sel(ID=longest).max()), "| of", events.attrs["N_events_final"], "events")
       print(list(events.data_vars))
       client.close()

The ``if __name__ == "__main__":`` guard is needed because the local cluster starts worker
processes that re-import the script. The ``memory_limit`` is passed explicitly because Dask does
not read a SLURM or container memory cap. Size both to your own allocation.

Output
======

``preprocess_data`` returns a Dataset with ``dat_anomaly``, ``mask``, ``extreme_events`` and
``thresholds``. The tracker returns ``ID_field`` (the event ID at every cell and time, 0 for
background), ``area``, ``centroid``, ``presence``, ``time_start``, ``time_end``, ``global_ID`` and
``merge_ledger``. On this synthetic input the longest-lived event is the planted 40-day patch (the run printed
40 days, starting 2015-01-10), alongside many short-lived events from the noise. Those are an artefact of 5 % exceedance on random data
and are removed in real use by the area filter and by examining duration.

Measured Scale
==============

On 40 years of daily 0.25 degree global SST (output 9282 days x 720 x 1440 after the 15-year
trim), approximate detection took 3954 s with ``persist`` and 3663 s with ``streaming``, on 4
workers x 16 threads x 22 GB on one compute node of DKRZ's Levante system. Both arms
produced identical arrays and neither restarted a worker. This is one run per arm, measured, not
guaranteed.

Caveats
=======

* Percentile thresholds define extremes relative to the local climate. A cell that is never
  warm enough to matter ecologically is still flagged at its own 95th percentile.
* The first ``window_years`` years of a ``shifting_baseline`` run are not available as
  output. For a record shorter than that, use ``fixed_baseline`` with a ``reference_period``.
* The tracker requires a bool field, a Dask ``Client`` and, for the default global
  coordinate handling, longitudes covering about 360 degrees. Regional domains use
  :func:`marEx.regional_tracker`.
* ``R_fill`` is in grid cells and ``T_fill`` is in time steps, so both depend on the grid
  and cadence. See :doc:`../guide/tracking`.
