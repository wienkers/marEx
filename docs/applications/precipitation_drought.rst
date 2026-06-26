=====================
Precipitation Drought
=====================


Meteorological drought is a persistent deficit of precipitation, and its consequences reach
hydropower reservoirs, irrigation supply and crop yield. A drought is characterised by the
months it lasts and the region it covers, not by a single dry month at one location.

Why marEx Fits
==============

* Monthly input is supported. The cadence is inferred from the time axis and thresholds are
  computed per calendar month (the ``thresholds`` variable gets a ``month`` dimension).
* ``tail="lower"`` flags the driest months for each cell and calendar month, so a normally
  dry season is judged against its own dry months.
* The tracker joins dry cells into spatially coherent events that persist across months,
  with a start, an end and an area at each time. This gives the extent and duration that
  matter for reservoir catchments and growing regions.

Configuration
=============

The snippet builds 30 years of monthly precipitation, plants an eight-month dry spell over a
block of cells, and flags the driest 10 % for each calendar month. Monthly data have no
21-day smoothing to apply, so ``smooth_days=1``.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import xarray as xr
   import marEx


   if __name__ == "__main__":

       rng = np.random.default_rng(3)
       time = pd.date_range("1990-01-01", "2019-12-31", freq="MS")   # 30 years, monthly
       lat = np.linspace(-10, 20, 20)
       lon = np.linspace(0, 90, 30)
       mon = time.month.values[:, None, None]
       pr = (
           80 + 60 * np.cos(2 * np.pi * (mon - 7) / 12)
           + 25 * rng.standard_normal((time.size, lat.size, lon.size))
       ).clip(min=0)
       pr[200:208, 5:14, 8:22] *= 0.2   # an eight-month dry spell
       da = xr.DataArray(pr.astype("float32"), dims=("time", "lat", "lon"),
                         coords={"time": time, "lat": lat, "lon": lon}, name="precip").chunk({"time": 60})

       ds = marEx.preprocess_data(
           da,
           method_anomaly="fixed_baseline",
           method_extreme="seasonal_percentile",   # thresholds per calendar month
           threshold_percentile=10,
           tail="lower",
           smooth_days=1,                          # a 21-day smoother has no meaning on monthly steps
           window_days=31,                         # pool the neighbouring months: one 31-day window = 1 step
           reference_period=(1990, 2019),
       )
       print(ds.thresholds.dims, int(ds.extreme_events.sum().compute()))

       client = marEx.helper.start_local_cluster(n_workers=2, threads_per_worker=2, memory_limit="3GB")
       events = marEx.regional_tracker(
           ds.extreme_events, ds.mask, coordinate_units="degrees",
           R_fill=2, T_fill=2, area_filter_absolute=30, grid_resolution=3.0,
       ).run()
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

``extreme_events`` marks dry months. The tracker output counts duration in the time axis of
the input, so on monthly data ``time_start`` and ``time_end`` are month starts and the
elapsed time between them is in months. ``T_fill`` is in time steps, which here are months.

Caveats
=======

* The anomaly is a departure from the climatology of the raw variable. Precipitation is
  skewed and bounded at zero, so anomalies of raw totals are not equally meaningful in wet and
  arid regions. Transform upstream (a standardised precipitation index, for example) if the
  application needs one.
* Monthly data cannot resolve a dry spell shorter than a month, and a drought that cares about
  the timing of rain within the growing season needs a finer cadence.
* The 10th percentile of a monthly record is estimated from one value per year. Thirty years
  is thirty samples per calendar month, before pooling neighbouring cells. Check the
  sample-count warning and consider ``window_spatial``.
* Seasonal percentiles flag dry months in every season, including those with little water
  to lose. A dry spell during the wet season is more consequential than one during the dry
  season, and the percentile does not weight that.
* Long monthly shifting baselines lose ``window_years`` years at the start. Here the baseline
  is a fixed reference period.
