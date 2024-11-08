ocetrac-dask
==============================

`Ocetrac-dask` is a Python 3.6+ package based on [ocetrac](https://github.com/ocetrac/ocetrac) which labels and tracks unique geospatial features from gridded datasets. This version has been rewritten to accept larger-than-memory spatio-temporal datasets and process them in parallel using [dask](https://dask.org/). It avoids loop-carried dependencies in time, keeps dask arrays distributed in memory throughout, and leverages the [dask-image](https://github.com/dask/dask-image) library. These modifications has allowed preliminary scaling to 40 years of _daily_ data on 1024 cores. 

These major modifications to support long daily timeseries of global 3D data at increasingly high spatial resolution has been necessitated by the [EERIE project](https://eerie-project.eu).

For `ocetrac-dask`-specific questions, please contact [Aaron Wienkers](mailto:aaron.wienkers@usys.ethz.ch)


Examples Notebooks with Dask
------------
1. ../notebooks/01_preprocess_dask.ipynb --- Preprocessing of data with Dask. 14000 daily 2D (0.25°) outputs processed in ~5 minutes on 128 cores.
2. ../notebooks/02_track_dask.ipynb --- Track MHWs using dask-powered ocetrac algorithm. 14000 daily 2D (0.25°) outputs processed in ~6 minutes on 128 cores.
3. ../notebooks/03_visualise_dask.ipynb --- Some dask-backed visualisation routines
   

Installation
------------

**PyPI**

To install the core package run: ``pip install git+https://github.com/wienkers/ocetrac-dask.git`` 

**GitHub**

1. Clone ocetrac to your local machine: ``git clone https://github.com/wienkers/ocetrac-dask.git``
2. Change to the parent directory of ocetrac
3. Install ocetrac with ``pip install -e ./ocetrac-dask``. This will allow
   changes you make locally, to be reflected when you import the package in Python


Future Work
------------
- [ ] Support for 3D MHW tracking