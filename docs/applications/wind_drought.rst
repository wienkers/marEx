============
Wind Drought
============


A wind drought is a spell of unusually low wind over a region where generation is
concentrated. A lull's extent and duration determine how much generation falls across the sites of a portfolio together, and whether diversification across a grid helps.

Why marEx Fits
==============

* ``tail="lower"`` with ``threshold_percentile=10`` flags
  values at or below the 10th percentile for each time of year.
* Tracked footprints give the quantities that matter for diversification: the area of a
  lull through time, how long it lives, and where its centroid moves.
* Hourly and 6-hourly input is thresholded per
  hour of the year. ``window_days`` and ``smooth_days`` are in days and converted to cycle
  slots and time steps.

Configuration
=============

The snippet builds 6 years of daily wind speed on a regional grid with a 15-day lull planted
over a block of cells. It uses ``fixed_baseline`` and the lower 10 % tail, then tracks the
lulls.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import xarray as xr
   import marEx


   if __name__ == "__main__":

       rng = np.random.default_rng(2)
       time = pd.date_range("2000-01-01", "2005-12-31", freq="D")
       lat = np.linspace(40, 65, 20)
       lon = np.linspace(-15, 30, 30)
       doy = time.dayofyear.values[:, None, None]
       ws = (
           7 + 2 * np.cos(2 * np.pi * (doy - 15) / 365.25)
           + 1.5 * rng.standard_normal((time.size, lat.size, lon.size))
       ).clip(min=0.1)
       ws[1200:1215, 4:14, 6:22] -= 4.0   # a 15-day regional lull
       ws = ws.clip(min=0.1)
       da = xr.DataArray(ws.astype("float32"), dims=("time", "lat", "lon"),
                         coords={"time": time, "lat": lat, "lon": lon}, name="wind_speed_100m").chunk({"time": 100})

       ds = marEx.preprocess_data(
           da,
           method_anomaly="fixed_baseline",      # keeps the full series and suits sub-daily data
           method_extreme="seasonal_percentile",
           threshold_percentile=10,
           tail="lower",                         # lowest 10 % of wind for the time of year
       )

       client = marEx.helper.start_local_cluster(n_workers=2, threads_per_worker=2, memory_limit="3GB")
       events = marEx.regional_tracker(
           ds.extreme_events, ds.mask, coordinate_units="degrees",
           R_fill=2, T_fill=2, area_filter_absolute=30, grid_resolution=1.5,
       ).run()

       dur = (events.time_end - events.time_start) / np.timedelta64(1, "D") + 1   # days
       longest = dur.idxmax("ID")                                                 # ID of the longest-lived event
       print("longest event:", int(longest), "|", float(dur.sel(ID=longest)), "days | starts",
             str(events.time_start.sel(ID=longest).values)[:10], "| peak area",
             float(events.area.sel(ID=longest).max()), "| of", events.attrs["N_events_final"], "events")
       client.close()

The ``if __name__ == "__main__":`` guard is needed because the local cluster starts worker
processes that re-import the script. Pass ``memory_limit`` explicitly and size the cluster to your
own allocation, since Dask sizes workers from the machine's physical memory.

On sub-daily input (hourly or 6-hourly wind), ``fixed_baseline`` and ``shifting_baseline``
both resolve their climatology by hour of year, so the diurnal cycle goes into the baseline
and is removed from the anomaly. See :doc:`../guide/dimensions_and_time`.

For a worked example on reanalysis data, see the ERA5 notebook in :doc:`../tutorials/index`.

Output
======

The tracker output gives, for each lull, ``time_start`` and ``time_end`` (duration), ``area``
through time, and the ``centroid`` track. ``ID_field`` marks the cells in each lull at each
time, so overlaying it on a map of installed capacity gives the capacity exposed by each
event. The overlay is plain xarray.

Caveats
=======

* Wind speed at hub height and capacity factor behave differently. Capacity factor is bounded
  by 0 and 1 and saturates at rated speed, so its lower tail can contain many tied values.
* A low-wind spell defined by percentile of wind speed is not necessarily a low-generation
  spell, because the power curve is nonlinear. If generation is the target, transform to
  capacity factor first.
* A seasonal lower-tail threshold flags the calmest 10 % of days for each time of year, so
  filter the catalogue by tracked duration and area for the lulls that matter to a system.
* Tracker parameters in time steps (``T_fill``) mean different durations on hourly and daily data.
