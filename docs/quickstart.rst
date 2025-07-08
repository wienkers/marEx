==========
Quickstart
==========

This guide will get you up and running with marEx in just a few minutes.

Basic Example
=============

Here's a simple example to detect and track marine heatwaves:

.. code-block:: python

   import xarray as xr
   import marEx

   # Load your sea surface temperature data
   sst = xr.open_dataset('your_sst_data.nc', chunks={}).sst

   # Detect extreme events (marine heatwaves)
   extremes = marEx.preprocess_data(sst, threshold_percentile=95)

   # Track events through time
   tracker = marEx.tracker(extremes.extreme_events, extremes.mask)
   tracked_events = tracker.run()

   # Visualise results
   fig, ax, im = (tracked_events.ID_field > 0).mean("time").plotX.single_plot(marEx.PlotConfig(var_units="MHW Frequency", cmap="hot_r", cperc=[0, 96]))

   # Create animated movie of tracked events
   tracked_events.ID_field.plotX.animate(marEx.PlotConfig(plot_IDs=True), plot_dir="./plots", file_name="mhw_animation")


That's it! You've detected, tracked, and visualized marine heatwaves in your data.

Next Steps
==========

* Read the :doc:`user_guide` for detailed workflows
* Explore the :doc:`api` for all available functions
* Check out the :doc:`examples` for more complex scenarios
