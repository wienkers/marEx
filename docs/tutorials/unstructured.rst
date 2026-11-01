=================
Unstructured Data
=================

Marine heatwaves detected and tracked in ICON-O ocean model output on its native R02B09 mesh (14.9 million cells). Detection takes the cell
dimension in place of lat/lon, tracking uses the mesh's neighbour table and cell
areas, and plotting needs the grid triangulation, registered with
:func:`marEx.specify_grid`. Work through the notebooks in order.

.. toctree::
   :maxdepth: 1

   1 · Detect extremes <unstructured/01_preprocess_extremes>
   2 · Identify & track events <unstructured/02_id_track_events>
   3 · Visualise & analyse <unstructured/03_visualise_events>
