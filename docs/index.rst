======================================================
marEx: Weather & Climate Extremes Detection & Tracking
======================================================

.. image:: https://github.com/wienkers/marEx/actions/workflows/ci.yml/badge.svg
   :target: https://github.com/wienkers/marEx/actions/workflows/ci.yml
   :alt: CI
.. image:: https://codecov.io/gh/wienkers/marEx/branch/main/graph/badge.svg
   :target: https://codecov.io/gh/wienkers/marEx
   :alt: codecov
.. image:: https://badge.fury.io/py/marEx.svg
   :target: https://badge.fury.io/py/marEx
   :alt: PyPI version
.. image:: https://static.pepy.tech/badge/marex
   :target: https://pepy.tech/projects/marex
   :alt: PyPI Downloads
.. image:: https://zenodo.org/badge/945834123.svg
   :target: https://doi.org/10.5281/zenodo.16922881
   :alt: DOI

**marEx** computes climatologies and anomalies, identifies extremes, and tracks
the resulting events through space and time in geospatial time series. Each of
its three modules has several methods for different applications and domains,
runs on data larger than memory, and scales with Dask from a laptop to a
distributed supercomputer. The same calls detect marine and atmospheric
heatwaves, cold spells, droughts and subsurface ocean extremes, on lat/lon grids
and unstructured meshes alike. On lat/lon grids, events can also be tracked as physical
volumes in 4-D, through ``(time, depth, lat, lon)``.

.. grid:: 1 1 2 2
   :gutter: 3
   :margin: 4 0 0 0

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-mr-1` Get started
      :link: getting_started/index
      :link-type: doc

      Install marEx and run a first anomaly, detection and tracking workflow.

   .. grid-item-card:: :octicon:`book;1.5em;sd-mr-1` Tutorials
      :link: tutorials/index
      :link-type: doc

      End-to-end notebooks on real data.

   .. grid-item-card:: :octicon:`globe;1.5em;sd-mr-1` Applications
      :link: applications/index
      :link-type: doc

      Marine and atmospheric heatwaves, wind and precipitation drought,
      subsurface ocean extremes and event catalogues.

   .. grid-item-card:: :octicon:`mortar-board;1.5em;sd-mr-1` User Guide
      :link: guide/index
      :link-type: doc

      Anomalies, extremes, dimensions and time, tracking, performance and
      validation.

   .. grid-item-card:: :octicon:`code;1.5em;sd-mr-1` API Reference
      :link: api/index
      :link-type: doc

      Every public function and class.

   .. grid-item-card:: :octicon:`megaphone;1.5em;sd-mr-1` What's new in 5.0
      :link: whats_new
      :link-type: doc

      Changes since 4.1 and how to migrate from 4.x.

Quick example
=============

.. code-block:: python

   import xarray as xr
   import marEx

   # In a script, run this under `if __name__ == "__main__":` (its worker processes re-import the script).
   client = marEx.helper.start_local_cluster(n_workers=4, memory_limit="8GB")

   # Load sea surface temperature (Dask-backed)
   sst = xr.open_dataset("sst_data.nc", chunks={"time": 25}).sst

   # 1 + 2. Anomalies, then extremes above the seasonal 95th percentile
   extremes = marEx.preprocess_data(
       sst,
       method_anomaly="shifting_baseline",
       method_extreme="seasonal_percentile",
       threshold_percentile=95,
   )

   # 3. Track events through time
   events = marEx.tracker(
       extremes.extreme_events, extremes.mask,
       R_fill=8, area_filter_quartile=0.5, allow_merging=True,
   ).run()

   # Visualise
   fig, ax, im = (events.ID_field > 0).mean("time").plotX.single_plot(
       marEx.PlotConfig(var_units="Event Frequency", cmap="hot_r", cperc=[0, 96])
   )

The three modules are ``marEx.anomaly``, ``marEx.extremes`` and
``marEx.tracker``, and each also runs on its own. ``preprocess_data`` chains the
first two.

Why marEx?
==========

* **Larger than memory**: ``compute_mode="streaming"`` stages intermediates on
  disk, so the events in 20 years of 6-hourly global SST (a 91 GB ``ID_field``)
  are tracked inside a 32 GiB job.
* **Overlap-based tracking**: merges and splits are decided by area overlap and
  recorded in ``merge_ledger``, so events that only touch keep their own
  identities.
* **4-D spatiotemporal event tracking**: declare depth and the tracker follows events through
  ``(time, depth, lat, lon)`` on lat/lon grids, with each event's volume in
  m\ :sup:`3` and depth range through time.
* **Any grid, cadence and tail**: one API for lat/lon grids and unstructured
  meshes, 4-D fields with depth or pressure level, daily, monthly and
  sub-daily data, and either tail of the distribution.

See :doc:`why_marex` for each point in detail.

.. toctree::
   :hidden:
   :caption: Getting Started

   getting_started/index

.. toctree::
   :hidden:
   :caption: Tutorials

   tutorials/index

.. toctree::
   :hidden:
   :caption: Applications

   applications/index

.. toctree::
   :hidden:
   :caption: User Guide

   guide/index

.. toctree::
   :hidden:
   :caption: Reference

   whats_new
   why_marex
   api/index
   troubleshooting
