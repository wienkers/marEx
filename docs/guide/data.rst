===================
Preparing Your Data
===================


Input-data requirements, loading patterns, and grid conventions for the
marEx pipeline. Once your data is loaded and Dask-backed, continue to
:doc:`detection`.


Data Preparation
================

Input Data Requirements
-----------------------

marEx works with xarray DataArrays containing oceanographic time series data:

**Required Dimensions:**

* **Time dimension**: Regular or irregular time steps (daily to monthly)
* **Spatial dimensions**: Either structured (lat, lon) or unstructured (cell/ncells)

**Supported Formats:**

* NetCDF (.nc)
* Zarr (.zarr)
* Any xarray-compatible format

**Data Structure Examples:**

For structured/gridded data::

   data.dims = ('time', 'lat', 'lon')
   # Coordinates: time, lat, lon as dimensions

For unstructured data::

   data.dims = ('time', 'ncells')
   # Coordinates: time as dimension, lat/lon as coordinates

Data Quality Requirements
-------------------------

* **Temporal coverage**: Minimum 10 years for robust climatology
* **Spatial coverage**: Consistent grid throughout time series

Data Loading Examples
---------------------

**Loading NetCDF files:**

.. code-block:: python

   import xarray as xr

   # Single file
   ds = xr.open_dataset('sst_data.nc', chunks={})
   sst = ds.sst

   # Multiple files
   ds = xr.open_mfdataset('sst_*.nc', parallel=True, chunks={})
   sst = ds.sst

**Loading Zarr datasets:**

.. code-block:: python

   # Local Zarr
   ds = xr.open_zarr('data.zarr', chunks={})

   # Cloud-optimised Zarr
   ds = xr.open_zarr('gs://bucket/data.zarr', chunks={})

**Optimising data loading:**

.. code-block:: python

   # Apply chunking for efficient processing (chunk sizes vary by operation)
   sst = sst.chunk({'time': 365, 'lat': 50, 'lon': 100})

   # Ensure Dask backing
   if not marEx.is_dask_collection(sst.data):
       sst = sst.chunk()

**Note**: Optimal chunk sizes depend on your dataset size and the operation (preprocessing vs tracking). See the Performance Optimisation section below for detailed chunking strategies.

Grid Types and Coordinate Systems
=================================

Structured/Gridded Data
-----------------------

Regular latitude-longitude grids are the most common data type:

.. code-block:: python

   # Typical structured data
   print(sst.dims)         # e.g. ('time', 'lat', 'lon')
   print(sst.coords)       # e.g. ('time', 'lat', 'lon')

   # marEx automatically detects structured grids
   processed = marEx.preprocess_data(sst)

Unstructured Data
-----------------

Irregular meshes common in coastal and global ocean models:

.. code-block:: python

   # Typical unstructured data (e.g., ICON model)
   print(sst.dims)         # e.g. ('time', 'ncells')
   print(sst.coords)       # e.g. ('time', 'lat', 'lon') as coordinates, not dims

   # Specify grid configuration
   marEx.specify_grid(
       grid_type='unstructured',
       fpath_tgrid='grid_info.nc',      # Grid topology file
       fpath_ckdtree='kdtree.pkl'       # Spatial index (optional)
   )

**Requirements:**

* Spatial dimension name (``ncells``, ``cell``, etc.)
* Latitude/longitude as coordinate arrays
* Optional: Grid topology information for advanced/unstructured features
