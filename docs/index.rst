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

**marEx** detects and tracks climate extremes on any grid, in datasets larger
than memory. It has three stages, each usable on its own: anomalies, extremes
and event tracking. It runs on lat/lon grids and unstructured meshes, on daily,
monthly or sub-daily data, on 2-D fields and fields with an extra dimension such
as depth, and on either tail of the distribution.

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

      End-to-end notebooks for gridded, regional and unstructured data.

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

      Changes since 4.1 and the migration table from the old names.

Quick example
=============

.. code-block:: python

   import xarray as xr
   import marEx

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

Each stage also runs alone. ``marEx.anomaly.compute`` returns anomalies with no
threshold anywhere, and ``marEx.extremes.identify`` thresholds anomalies from
any source.

Why marEx?
==========

* **Stages that stand alone**: use the anomaly stage, the extreme stage or the
  tracker independently.
* **Any grid, cadence and tail**: one API for lat/lon grids and unstructured
  meshes, extra dimensions, monthly and sub-daily data, and ``tail="lower"``.
* **Larger than memory**: ``compute_mode="streaming"`` keeps intermediates on
  disk. In a measured case, ``persist`` was OOM-killed in a 24 GiB allocation
  while ``streaming`` completed in the same allocation.
* **Overlap-based tracking**: merges and splits require overlap and are recorded
  in ``merge_ledger``, which avoids the spurious "mega-events" of naive 3-D
  connected-component labelling.
* **Chunking-independent results**: detection output does not depend on the
  input chunking, verified on the test fixtures for every anomaly and threshold
  method.

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
