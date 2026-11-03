================
Subsurface Ocean
================


Marine heatwaves are often described at the surface, but the water that matters for
many fisheries and for aquaculture sits at depth. A warm anomaly at 25 to 50 m can
be absent at the surface, and cages, benthic habitats and thermoclines sit at specific levels.

Why marEx Fits
==============

* A field ``(time, depth, lat, lon)`` is detected
  whole. Any dimension that is not time or horizontal is treated as an extra dimension,
  with a separate threshold at every level.
* Levels below the sea floor are NaN at the first time step and
  are masked at that point, so a shelf and an abyssal plain share one array.
* ``window_spatial`` pools neighbouring cells within each level.

Configuration
=============

The snippet builds a four-level temperature field with a warm anomaly confined to 25 and 50 m.
It runs detection on all levels at once, then selects one level for a 3-D track. Tracking the
anomaly as a 4-D event is covered in `Tracking in 4-D`_.

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

       # Per-level exceedance days. A 3-D track and plotX take one level, so select it first
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
processes that re-import the script. Pass ``memory_limit`` explicitly and size the cluster to your
own allocation, since Dask sizes workers from the machine's physical memory.

Output
======

The detection outputs keep the depth axis: ``dat_anomaly`` and ``extreme_events`` are
``(time, depth, lat, lon)``, ``mask`` is ``(depth, lat, lon)`` and ``thresholds`` is
``(depth, lat, lon, dayofyear)``. The ``plotX`` accessor takes one map at a time, so select a level
with ``.sel`` or loop over levels. The tracker works on either: a selected level gives 3-D
events, and the whole field with depth declared gives 4-D events.

Tracking in 4-D
===============

A warm anomaly that spans 25 to 50 m should be one event. Declare depth
under ``"z"`` in ``dimensions`` and pass the layer interfaces, and the tracker links
``extreme_events`` through ``(time, depth, lat, lon)`` with a volume in m\ :sup:`3` for every
event and timestep, plus its depth centroid and its top and bottom depth. The tracker also needs
a boolean wet mask ``(depth, lat, lon)`` that is ``True`` in the ocean, and either
``grid_resolution`` or ``cell_areas``:

.. code-block:: python

   events = marEx.tracker(
       ds.extreme_events,
       ds.mask,
       R_fill=2,
       dimensions={"time": "time", "z": "depth", "y": "lat", "x": "lon"},
       layer_bounds=layer_bounds,       # (depth, 2) interfaces in metres
       grid_resolution=15.0,
       nn_partitioning=True,
       volume_filter_quartile=0.5,
       compute_mode="streaming",
       temp_dir="/path/to/scratch/marex_staging",
   ).run()

The guide covers the vertical geometry, the gap rule and the memory model in :ref:`tracking-in-4d`.
For a worked run on a 36-level ocean reanalysis, see the notebooks in ``examples/4d/``
(:doc:`../tutorials/4d`).

Caveats
=======

* Tracking a selected level gives one event per level. Declare depth under ``"z"`` to track the
  anomaly as a single 4-D event.
* Detection needs finite values at every unmasked cell and time, so interpolate or mask
  data gaps first.
* The mask is taken from the first time step. A cell that is NaN there is treated as land or
  sea floor for the whole record.
* Vertical resolution of the input limits what depth structure can be resolved.
