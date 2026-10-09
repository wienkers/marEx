============
Applications
============

marEx handles any scalar field on a regular or unstructured grid, so the same three stages
(anomaly, extreme, track) serve several fields of study. Each page below states the problem,
what marEx does for it, a configuration that runs on a small synthetic field, the outputs, and
the caveats. The snippets use synthetic data. They show the API and the output structure, and
they are not results.

.. list-table::
   :widths: 28 72

   * - :doc:`marine_heatwaves`
     - Seasonal percentile thresholds on a shifting baseline, tracked as events. Reef,
       kelp and aquaculture exposure depends on footprint and persistence.
   * - :doc:`atmospheric_heatwaves`
     - Hot spells in 2 m temperature or a heat-stress index, with yearly exceedance days and
       a daily footprint. Heat stress and electricity demand peaks.
   * - :doc:`wind_drought`
     - Lower-tail wind speed or capacity factor, tracked. Extent and persistence of lulls
       decide how far sites of a portfolio fail together.
   * - :doc:`precipitation_drought`
     - Lower-tail monthly precipitation. Extent and duration of dry spells for hydropower
       catchments and growing regions.
   * - :doc:`subsurface_ocean`
     - Three-dimensional (time, depth, lat, lon) anomalies with per-level thresholds.
       Relevant to fisheries and aquaculture at depth.
   * - :doc:`climatologies_only`
     - The anomaly stage alone, on large datasets, with no threshold.
   * - :doc:`event_catalogues`
     - Tracked events as a table of duration, area, footprint and intensity, for frequency and
       severity distributions and clustering. Structurally an event set.

Choosing a Configuration
========================

.. list-table::
   :header-rows: 1
   :widths: 28 24 24 24

   * - Case
     - Anomaly method
     - Threshold
     - Tail
   * - Marine heatwave
     - ``shifting_baseline``
     - ``seasonal_percentile``
     - upper
   * - Atmospheric heatwave
     - ``fixed_baseline``
     - ``seasonal_percentile``
     - upper
   * - Wind drought
     - ``fixed_baseline``
     - ``seasonal_percentile``
     - lower
   * - Precipitation drought
     - ``fixed_baseline``
     - ``seasonal_percentile``
     - lower
   * - Subsurface ocean
     - ``fixed_baseline``
     - ``seasonal_percentile``
     - upper

These are the configurations in the snippets and not prescriptions. The guide pages
:doc:`../guide/anomalies` and :doc:`../guide/extremes` give the full option tables, and
:doc:`../guide/tracking` covers the tracker parameters.

.. toctree::
   :maxdepth: 1
   :hidden:

   marine_heatwaves
   atmospheric_heatwaves
   wind_drought
   precipitation_drought
   subsurface_ocean
   climatologies_only
   event_catalogues
