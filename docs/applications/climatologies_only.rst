==================
Climatologies Only
==================


The anomaly stage has no threshold. Anomalies of a large dataset are often the result
itself: input to an empirical orthogonal function analysis, a regression target, a training
set or a seasonal-cycle diagnostic. ``marEx.anomaly.compute`` runs that stage alone.

Why marEx Fits
==============

* Four anomaly methods cover the common baselines: ``shifting_baseline`` (rolling
  climatology), ``fixed_baseline`` (a fixed reference period), ``detrend_fixed_baseline``
  and ``detrend_harmonic`` (harmonic seasonal cycle plus polynomial trend, not available on sub-daily data).
* The climatology is computed lazily on a Dask array, so the input is read chunk by chunk. With ``compute_mode="streaming"`` the intermediate fields go to a Zarr store in
  ``scratch_dir`` instead of staying pinned in cluster memory.
* The same code handles gridded and unstructured grids, 3-D fields with an extra
  dimension, and non-daily cadences.

Configuration
=============

The snippet computes a shifting-baseline anomaly, a fixed-baseline anomaly with a reference
period, and a streaming run that writes the result and then clears the staging directory.

.. code-block:: python

   import numpy as np
   import pandas as pd
   import xarray as xr
   import marEx

   rng = np.random.default_rng(5)
   time = pd.date_range("2000-01-01", "2007-12-31", freq="D")
   lat = np.linspace(-60, 60, 20)
   lon = np.linspace(0, 348, 30)
   doy = time.dayofyear.values[:, None, None]
   sst = (15 + 8 * np.cos(2 * np.pi * (doy - 30) / 365.25)
          + 0.5 * rng.standard_normal((time.size, lat.size, lon.size)))
   da = xr.DataArray(sst.astype("float32"), dims=("time", "lat", "lon"),
                     coords={"time": time, "lat": lat, "lon": lon}, name="sst").chunk({"time": 100})

   # Anomaly stage alone: rolling 15-year climatology (3 years here), no threshold
   anom = marEx.anomaly.compute(da, method="shifting_baseline", window_years=3, smooth_days=21)
   print(anom.dat_anomaly.sizes, list(anom.data_vars))
   print(float(anom.dat_anomaly.mean().compute()))

   # Fixed 1-Jan-2000 .. 2003 reference climatology
   anom_fixed = marEx.anomaly.compute(da, method="fixed_baseline", reference_period=(2000, 2003))
   print(anom_fixed.dat_anomaly.sizes)

   # Streaming: write, then clear the staging directory
   import tempfile, os
   with tempfile.TemporaryDirectory() as scratch:
       out = marEx.anomaly.compute(da, method="fixed_baseline", compute_mode="streaming", scratch_dir=scratch)
       out.to_zarr(os.path.join(scratch, "anom.zarr"), mode="w")
       marEx.clear_staging(out)
   print("ok")

Output
======

A Dataset with ``dat_anomaly`` and ``mask``. With ``standardise=True`` (``detrend_harmonic``
only) it also has ``dat_stn`` and ``STD``. ``shifting_baseline`` returns a series shorter
by ``window_years`` years, since the first years lack a full baseline. The other methods keep
the full series.

With ``compute_mode="streaming"`` the returned Dataset reads lazily from a staging directory.
Write the output first, then call ``marEx.clear_staging(ds)``. The path is in
``ds.encoding["marex_staging_dir"]``. See :doc:`../guide/performance`.

Caveats
=======

* The input must be Dask-backed.
* The first time step defines the mask, and ``validate=True`` rejects NaN or infinite values
  at unmasked cells.
* ``shifting_baseline`` raises an error if the record spans fewer years than ``window_years``.
* ``reference_period`` applies only to the fixed-baseline methods.
* Anomalies are float32.
