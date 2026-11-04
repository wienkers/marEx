=========
Tutorials
=========

Jupyter notebooks running the **detect → track → visualise** workflow on real data, with the same calls throughout and a few grid-specific arguments.

.. grid:: 1 1 2 2
   :gutter: 3

   .. grid-item-card:: Gridded data
      :link: gridded
      :link-type: doc

      Global daily sea surface temperature (OSTIA, 1982-2022) on a regular
      lat/lon grid: marine heatwaves detected two ways, then tracked.

   .. grid-item-card:: Regional data
      :link: regional
      :link-type: doc

      The same workflow on a bounded European domain at 0.05°.
      :func:`marEx.regional_tracker` handles the non-periodic edges.

   .. grid-item-card:: Unstructured data
      :link: unstructured
      :link-type: doc

      ICON-O ocean model output detected and tracked on its native R02B09 mesh (14.9 million cells).

   .. grid-item-card:: European wind droughts
      :link: wind_drought
      :link-type: doc

      ERA5 wind over Europe, 1980-2024: lower-tail extremes of hub-height
      capacity factor tracked into a catalogue of low-wind spells.

   .. grid-item-card:: Ocean extremes in 4-D
      :link: 4d
      :link-type: doc

      GLORYS12 northeast Pacific marine heatwaves, 1993-2022 on 36 levels
      to 1150 m, tracked through time and depth as volumes.

Scripts
=======

`examples/batch_jobs <https://github.com/wienkers/marEx/tree/main/examples/batch_jobs>`_ is a template for running
detection and tracking as SLURM jobs: two driver scripts configured by one YAML file per dataset, with
worked configs for the gridded, regional and unstructured examples.
`examples/larger-than-memory <https://github.com/wienkers/marEx/tree/main/examples/larger-than-memory>`_ detects and
tracks 20 years of 6-hourly global SST inside one 32 GiB job and compares the two compute modes. It is the source of the 6-hourly rows
in :doc:`../guide/performance`. An index of every example, with its input and measured run
time, is in `examples/README.md <https://github.com/wienkers/marEx/blob/main/examples/README.md>`_.

.. toctree::
   :hidden:

   gridded
   regional
   unstructured
   wind_drought
   4d
