================
Subsurface Ocean
================


Marine heatwaves are often described at the surface, but the water that matters for
many fisheries and for aquaculture sits at depth. A warm anomaly at 25 to 50 m can
be absent at the surface, and cages, benthic habitats and thermoclines sit at specific levels.

Why marEx Fits
==============

* Extra dimensions are kept. A field ``(time, depth, lat, lon)`` is detected
  as it is. Any dimension that is not time or horizontal is treated as an extra dimension,
  with a separate threshold at every level.
* The mask is three-dimensional. Levels below the sea floor are NaN at the first time step and
  are masked at that point, so a shelf and an abyssal plain share one array.
* Pooling is horizontal only. ``window_spatial`` pools neighbouring cells within a level and
  never across depth.

Configuration
=============

The snippet builds a four-level temperature field with a warm anomaly confined to 25 and 50 m.
It runs the detection stage on all levels at once, then selects one level for tracking.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import xarray as xr
   import marEx


   if __name__ == "__main__":

       rng = np.random.default_rng(4)
       time = pd.date_range("2000-01-01", "2005-12-31", freq="D")
       depth = np.array([5.0, 25.0, 50.0, 100.0])
       lat = np.linspace(-40, 40, 16)
       lon = np.linspace(0, 348, 24)
       doy = time.dayofyear.values[:, None, None, None]
       temp = (
           18 - 0.04 * depth[None, :, None, None]
           + 3 * np.cos(2 * np.pi * (doy - 40) / 365.25) * np.exp(-depth[None, :, None, None] / 60)
           + 0.4 * rng.standard_normal((time.size, depth.size, lat.size, lon.size))
       )
       temp[1000:1030, 1:3, 4:9, 6:14] += 2.5     # warm anomaly confined to 25-50 m
       da = xr.DataArray(temp.astype("float32"), dims=("time", "depth", "lat", "lon"),
                         coords={"time": time, "depth": depth, "lat": lat, "lon": lon},
                         name="thetao").chunk({"time": 100})

       ds = marEx.preprocess_data(da, method_anomaly="fixed_baseline", method_extreme="seasonal_percentile",
                                  threshold_percentile=95)
       print(ds.dat_anomaly.dims, ds.thresholds.dims, ds.mask.dims)

       # Per-level exceedance days. The tracker and plotX are 2-D, so select a level first
       days = ds.extreme_events.sum("time").compute()
       print(days.mean(["lat", "lon"]).values)

       client = marEx.helper.start_local_cluster(n_workers=2, threads_per_worker=2, memory_limit="3GB")
       lvl = ds.sel(depth=25.0)
       events = marEx.tracker(lvl.extreme_events, lvl.mask, R_fill=2, area_filter_absolute=15, grid_resolution=15.0).run()
       dur = (events.time_end - events.time_start) / np.timedelta64(1, "D") + 1   # days
       longest = dur.idxmax("ID")                                                 # ID of the longest-lived event
       print("longest event:", int(longest), "|", float(dur.sel(ID=longest)), "days | starts",
             str(events.time_start.sel(ID=longest).values)[:10], "| peak area",
             float(events.area.sel(ID=longest).max()), "| of", events.attrs["N_events_final"], "events")
       client.close()

The ``if __name__ == "__main__":`` guard is needed because the local cluster starts worker
processes that re-import the script. The ``memory_limit`` is passed explicitly because Dask does
not read a SLURM or container memory cap. Size both to your own allocation.

Output
======

The detection outputs keep the depth axis: ``dat_anomaly`` and ``extreme_events`` are
``(time, depth, lat, lon)``, ``mask`` is ``(depth, lat, lon)`` and ``thresholds`` is
``(depth, lat, lon, dayofyear)``. The tracker and the ``plotX`` plotting accessor are 2-D and
reject extra dimensions, so select a level with ``.sel`` or loop over levels.

Caveats
=======

* Events are tracked within a level. The tracker does not join events across depth, so a
  warm anomaly spanning several levels appears as one event per level. Matching them is up
  to the application.
* Detection needs finite values at every unmasked cell and time, so interpolate or mask
  data gaps first.
* The mask is taken from the first time step. A cell that is NaN there is treated as land or
  sea floor for the whole record.
* Vertical resolution of the input limits what depth structure can be resolved, and the
  thermocline is only as well represented as the source model or analysis represents it.
