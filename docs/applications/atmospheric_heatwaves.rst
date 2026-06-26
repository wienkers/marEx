=====================
Atmospheric Heatwaves
=====================


Atmospheric heatwaves matter to people through heat stress and through electricity demand,
and both depend on how widely a hot spell spreads as well as how hot it gets. Cooling load
rises across a whole system when a hot spell covers it, and heat-related health impacts
scale with the population under the footprint.

Why marEx Fits
==============

* The field is whatever scalar you supply. Daily maximum 2 m temperature, a wet-bulb
  temperature, a heat index or cooling degree days all pass through the same stages, because
  the stages do not assume a variable.
* The anomaly and the threshold are separate steps. :func:`marEx.anomaly.compute` followed by
  :func:`marEx.extremes.identify` lets you choose the anomaly definition (a fixed reference
  period here, to match an operational baseline) independently of the exceedance rule.
* ``seasonal_percentile`` flags a day that is hot for its time of year. ``global_percentile``
  uses one threshold per cell over all days, which is closer to an absolute-magnitude rule
  and flags mostly summer days.
* Exceedance fields give footprint directly. The fraction of valid cells exceeding on each
  day is a footprint time series, and the tracker gives each spell an identity, an area and a
  centroid track.

Configuration
=============

The snippet builds 6 years of daily 2 m temperature on a regional 20 x 30 grid, with a
two-week heatwave planted in year 5. It computes anomalies against the full record, flags the
upper 5 % per day of year, derives yearly exceedance days and a daily footprint fraction, and
tracks the spells.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import xarray as xr
   import marEx


   if __name__ == "__main__":

       rng = np.random.default_rng(1)
       time = pd.date_range("2000-01-01", "2005-12-31", freq="D")
       lat = np.linspace(35, 70, 20)
       lon = np.linspace(-10, 40, 30)
       doy = time.dayofyear.values[:, None, None]
       t2m = (
           8 + 12 * np.cos(2 * np.pi * (doy - 200) / 365.25)
           + 3 * rng.standard_normal((time.size, lat.size, lon.size))
       )
       t2m[1650:1664, 5:12, 8:20] += 7.0  # a two-week heatwave
       da = xr.DataArray(t2m.astype("float32"), dims=("time", "lat", "lon"),
                         coords={"time": time, "lat": lat, "lon": lon}, name="t2m").chunk({"time": 100})

       anom = marEx.anomaly.compute(da, method="fixed_baseline", smooth_days=21)
       ds = marEx.extremes.identify(anom, method="seasonal_percentile", threshold_percentile=95, tail="upper")

       # Exceedance days per cell and per year, and the footprint (fraction of cells exceeding) per day
       mask = ds.mask
       days_per_year = ds.extreme_events.groupby("time.year").sum("time").where(mask)
       footprint = ds.extreme_events.where(mask).mean(["lat", "lon"])
       print(days_per_year.compute().sizes, float(footprint.max().compute()))

       # Regional footprint tracking requires a Client and ~360-degree coordinates, or the regional tracker
       client = marEx.helper.start_local_cluster(n_workers=2, threads_per_worker=2, memory_limit="3GB")
       events = marEx.regional_tracker(
           ds.extreme_events, ds.mask, coordinate_units="degrees", R_fill=2, area_filter_absolute=20, T_fill=2
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

``days_per_year`` is a ``(year, lat, lon)`` count of exceedance days, the usual input to a
heat-stress exposure map. ``footprint`` is the share of valid cells exceeding on each day.
The tracker output has the same structure as in :doc:`marine_heatwaves`. On a regional domain
the tracker call is :func:`marEx.regional_tracker` with ``coordinate_units``, because the
default tracker detects units from a roughly 360 degree longitude range.

Caveats
=======

* A percentile threshold is relative to local climate. It does not say whether a temperature
  is dangerous or whether it moves demand. For an impact threshold, compute the impact
  variable first (cooling degree days, for example) and then pass that to marEx.
* Electricity demand depends on calendar, behaviour and installed cooling as well as
  temperature. marEx identifies the hot spells. It does not model demand.
* Thresholds need enough samples. With a short record the sample-count warning fires, and
  ``window_days`` and ``window_spatial`` can be widened to pool more data.
* The tracker call here uses ``area_filter_absolute`` in cell counts because no
  ``grid_resolution`` is given. Areas are then in cells.
