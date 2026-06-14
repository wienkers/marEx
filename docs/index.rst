=============================================
marEx — Weather & Climate Extremes Detection & Tracking
=============================================

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

**marEx** is a high-performance Python framework for identifying and tracking
extremes — in the ocean, the atmosphere, or on land — in massive climate datasets.
It provides a complete, grid-agnostic pipeline from raw data preprocessing to
tracked-event visualisation, and scales identically from a laptop to a
1000-core supercomputer.

.. grid:: 1 1 2 2
   :gutter: 3
   :margin: 4 0 0 0

   .. grid-item-card:: :octicon:`rocket;1.5em;sd-mr-1` Get started
      :link: getting_started/index
      :link-type: doc

      Install marEx and run your first detect-track-visualise workflow in five
      minutes.

   .. grid-item-card:: :octicon:`book;1.5em;sd-mr-1` Tutorials
      :link: tutorials/index
      :link-type: doc

      End-to-end notebooks for gridded, regional, and unstructured data.

   .. grid-item-card:: :octicon:`mortar-board;1.5em;sd-mr-1` User Guide
      :link: guide/index
      :link-type: doc

      Concepts, method selection, parameter tuning, and performance.

   .. grid-item-card:: :octicon:`code;1.5em;sd-mr-1` API Reference
      :link: api/index
      :link-type: doc

      Complete reference for every public function and class.

Quick example
=============

.. code-block:: python

   import xarray as xr
   import marEx

   # Load sea surface temperature (Dask-backed)
   sst = xr.open_dataset("sst_data.nc", chunks={"time": 30}).sst

   # 1. Detect extremes
   extremes = marEx.preprocess_data(
       sst, threshold_percentile=95,
       method_anomaly="shifting_baseline", method_extreme="seasonal_percentile",
   )

   # 2. Track events through time
   events = marEx.tracker(
       extremes.extreme_events, extremes.mask,
       R_fill=8, area_filter_quartile=0.5, allow_merging=True,
   ).run()

   # 3. Visualise
   fig, ax, im = (events.ID_field > 0).mean("time").plotX.single_plot(
       marEx.PlotConfig(var_units="MHW Frequency", cmap="hot_r", cperc=[0, 96])
   )

Why marEx?
==========

* **Grid-agnostic** — the same API works for structured (lat/lon) and
  unstructured (FESOM/ICON/MPAS) grids.
* **Built for scale** — a Dask-first architecture processes datasets
  100–1000× larger than available RAM.
* **Scientifically rigorous** — four anomaly methods and a generalised
  Hobday extreme definition with spatial pooling.
* **Advanced tracking** — overlap-thresholded merge/split handling avoids the
  spurious "mega-events" of naive 3D connected-component methods.

See :doc:`why_marex` for the full comparison.

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
   :caption: User Guide

   guide/index

.. toctree::
   :hidden:
   :caption: Reference

   why_marex
   api/index
   troubleshooting
