=========
Tutorials
=========

End-to-end Jupyter notebooks demonstrating the full marEx workflow —
**preprocess → track → visualise** — for each supported grid type. The same
high-level API is used throughout; only the input data differs.

.. note::

   These notebooks are rendered from their committed outputs. They read datasets
   stored on HPC scratch storage that are not bundled with the documentation, so
   they are **not** re-executed during the docs build. Each page links back to
   the original notebook on GitHub, where you can download and adapt it.

.. grid:: 1 1 3 3
   :gutter: 3

   .. grid-item-card:: Gridded data
      :link: gridded
      :link-type: doc

      Regular lat/lon grids — satellite products (e.g. NOAA OISST) and climate
      models (e.g. CMIP6).

   .. grid-item-card:: Regional data
      :link: regional
      :link-type: doc

      Spatially bounded, higher-resolution domains with boundary handling.

   .. grid-item-card:: Unstructured data
      :link: unstructured
      :link-type: doc

      Irregular meshes from ocean models (FESOM, ICON-O, MPAS-Ocean).

.. toctree::
   :hidden:

   gridded
   regional
   unstructured
