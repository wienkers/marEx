=========
Tutorials
=========

End-to-end Jupyter notebooks, each running the full **detect → track → visualise**
workflow on real data. The calls are the same throughout, and only the input and
a few grid-specific arguments change.

.. note::

   These pages are rendered from the notebooks' committed outputs. The notebooks
   read datasets on HPC storage that are not bundled with the documentation, so
   they are not re-executed during the docs build. Each page links to the original
   notebook on GitHub, and each notebook reports the wall time and memory of every
   stage on the cluster it ran on.

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

      The same workflow on a bounded European domain at 0.05°, with
      :func:`marEx.regional_tracker` handling the non-periodic edges.

   .. grid-item-card:: Unstructured data
      :link: unstructured
      :link-type: doc

      ICON-O ocean model output on its native R02B09 mesh (14.9 million cells),
      detected and tracked without regridding.

   .. grid-item-card:: European wind droughts
      :link: wind_drought
      :link-type: doc

      ERA5 wind over Europe, 1980-2024: lower-tail extremes of hub-height
      capacity factor tracked into a catalogue of low-wind spells.

   .. grid-item-card:: Ocean extremes in 3-D
      :link: 3d
      :link-type: doc

      A century of daily North Atlantic temperature on seven levels down to
      1415 m: marine heatwaves at every depth with a shifting baseline.

Scripts
=======

`examples/batch_jobs <https://github.com/wienkers/marEx/tree/main/examples/batch_jobs>`_ holds the gridded detect and
track notebooks as plain Python scripts with SLURM wrappers, for running the same workflow without Jupyter.
`examples/larger-than-memory <https://github.com/wienkers/marEx/tree/main/examples/larger-than-memory>`_ holds the
measurement harness behind :doc:`../guide/performance`. An index of every example, with its input and measured run
time, is in `examples/README.md <https://github.com/wienkers/marEx/blob/main/examples/README.md>`_.

.. toctree::
   :hidden:

   gridded
   regional
   unstructured
   wind_drought
   3d
