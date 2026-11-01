=======================
European Wind Droughts
=======================

A worked energy application on 45 years of ERA5. Hourly 10 m wind is converted to
the capacity factor of a generic turbine at 100 m hub height, daily lower-tail
extremes are detected against the 1991-2020 normal, and the tracker joins them into
low-wind spells with a footprint, a duration and a total deficit. The notebook ends
with an event catalogue and the anatomy of the most severe spell. The script that
prepares the input from ERA5 is next to the notebook in ``examples/applications/``.
See :doc:`../applications/wind_drought` for the method in brief.

.. toctree::
   :maxdepth: 1

   Wind droughts in ERA5 <applications/wind_drought_europe>
