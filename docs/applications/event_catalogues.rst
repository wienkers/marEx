================
Event Catalogues
================


Tracked extremes make a table of events. Each has a start, an end, a footprint, a path and an
intensity, and a table of that form supports the questions that scalar indices cannot answer:
how often events of a given size occur, how severe the worst ones are, and whether they cluster
in time. This page assembles that table from the tracker output and the anomaly field.

Why marEx Fits
==============

* The event is a coherent object. The tracker gives each connected, persistent exceedance an
  ID that persists through splits and merges, so counts and sizes are of events and not of
  exceeding cells or days.
* The output has the pieces of a hazard record. ``time_start`` and ``time_end`` give duration,
  ``area`` gives extent through time, ``centroid`` gives the path, and ``ID_field`` gives the
  footprint. Intensity comes from the anomaly field sampled where ``ID_field`` marks the event.
* The record is reproducible. The definitions are a handful of parameters (percentile, fill
  radii, overlap, size filter) that can be varied to see how the catalogue depends on them.

Configuration
=============

The snippet detects and tracks events in a synthetic SST record, keeps events lasting at least 5
days, and builds a catalogue with duration, peak area, footprint size (the union of cells over
time), peak and mean anomaly, and mean centroid latitude. It then computes annual frequency and
an empirical rate of events lasting at least a given duration.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import xarray as xr
   import marEx


   if __name__ == "__main__":

       rng = np.random.default_rng(6)
       time = pd.date_range("1990-01-01", "2019-12-31", freq="D")
       lat = np.linspace(-60, 60, 20)
       lon = np.linspace(0, 348, 30)
       doy = time.dayofyear.values[:, None, None]
       sst = (15 + 8 * np.cos(2 * np.pi * (doy - 30) / 365.25)
              + 0.5 * rng.standard_normal((time.size, lat.size, lon.size)))
       for t0, y0, x0, amp in [(7600, 5, 5, 3.0), (8700, 10, 18, 4.0), (9500, 4, 22, 3.5)]:
           sst[t0:t0 + 30, y0:y0 + 5, x0:x0 + 7] += amp
       da = xr.DataArray(sst.astype("float32"), dims=("time", "lat", "lon"),
                         coords={"time": time, "lat": lat, "lon": lon}, name="sst").chunk({"time": 100})

       client = marEx.helper.start_local_cluster(n_workers=2, threads_per_worker=2, memory_limit="3GB")
       ds = marEx.preprocess_data(da, method_anomaly="shifting_baseline", threshold_percentile=95)
       events = marEx.tracker(ds.extreme_events, ds.mask, R_fill=2, area_filter_absolute=20, grid_resolution=10.0).run()
       events = events.compute()
       anomaly = ds.dat_anomaly.compute()
       n_years = ds.time.size / 365.25

       duration = (events.time_end - events.time_start) / np.timedelta64(1, "D") + 1   # days (daily data)
       max_area = events.area.max("time")                                              # km^2, from grid_resolution
       keep = events.ID.where(duration >= 5, drop=True).values                         # drop short-lived noise

       rows = []
       for i in keep:
           inside = events.ID_field == i                       # (time, lat, lon): cells belonging to event i
           footprint = inside.any("time")                      # union of the event over time
           lat_c = events.centroid.sel(ID=i, component=0).where(events.presence.sel(ID=i))
           rows.append(dict(
               ID=int(i),
               start=events.time_start.sel(ID=i).values,
               duration_days=float(duration.sel(ID=i)),
               max_area_km2=float(max_area.sel(ID=i)),
               footprint_cells=int(footprint.sum()),
               peak_anomaly=float(anomaly.where(inside).max()),
               mean_anomaly=float(anomaly.where(inside).mean()),
               centroid_lat_mean=float(lat_c.mean()),
           ))
       catalogue = pd.DataFrame(rows).set_index("ID")
       print(catalogue.round(2).to_string())

       # Annual frequency, and an empirical exceedance-rate curve for duration (events per year lasting >= d days)
       d = np.sort(catalogue.duration_days.values)[::-1]
       rate = np.arange(1, d.size + 1) / n_years
       print("events/yr:", round(d.size / n_years, 2))
       print(list(zip(d, rate.round(3))))
       client.close()

The ``if __name__ == "__main__":`` guard is needed because the local cluster starts worker
processes that re-import the script. The ``memory_limit`` is passed explicitly because Dask does
not read a SLURM or container memory cap. Size both to your own allocation.

Output
======

The catalogue is a ``pandas.DataFrame`` indexed by event ID. From it, frequency per year is a
count per year, severity distributions are quantiles of peak anomaly, area or duration, and the
empirical rate in the last lines is a count of events at least as long as each duration, divided
by the record length in years. Clustering in time can be examined from the inter-arrival times
of ``start``, or from year-to-year dispersion of the annual counts compared with a Poisson
expectation.

Relation to Catastrophe-Model Event Sets
========================================

A catastrophe model's hazard module is, in structure, an event set: a list of events, each with
a footprint and intensity, and an annual rate. A catalogue built here has the same structure,
with the difference that it is derived from one observed, reanalysed or simulated record instead
of being generated stochastically. It describes hazard only. Vulnerability, exposure and loss
are outside marEx. Applying the same pipeline to each member of a simulation ensemble is a
natural way to enlarge the sample, and because the tracker rejects extra dimensions, members are
processed one at a time.

Caveats
=======

* Counts depend on the parameters. ``R_fill``, ``T_fill``, ``overlap_threshold`` and the size
  filter all change what counts as one event, so vary them to see how stable the quantity of
  interest is.
* Return periods from a short record are extrapolations. A catalogue of a few dozen events
  supports empirical rates at short durations, not a tail estimate. marEx does not fit extreme
  value distributions.
* Trends and decadal variability break stationarity. The shifting baseline removes the
  mean trend from the anomaly but not changes in event frequency.
* ``ID_field`` is a lazy ``(time, lat, lon)`` array. The snippet computes it in memory, which
  suits small fields. For large records, compute per event or window, or use
  ``compute_mode="streaming"`` in the tracker and process the output in chunks.
