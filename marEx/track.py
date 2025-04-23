"""
MarEx: Marine Extreme Event Identification, Tracking, and Splitting/Merging Module

MarEx identifies and tracks extreme events in oceanographic data across time, 
supporting both structured (regular grid) and unstructured datasets. It can identify
discrete objects at single time points and track them as evolving events through time,
seamlessly handling splitting and merging.

This package provides algorithms to:
- Identify binary objects in spatial data at each time step
- Track these objects across time to form coherent events
- Handle merging and splitting of objects over time
- Calculate and maintain object/event properties through time
- Filter by size criteria to focus on significant events

Key terminology:
- Object: A connected region in binary data at a single time point
- Event: One or more objects tracked through time and identified as the same entity
"""

import xarray as xr
import numpy as np
from dask.distributed import wait
from dask_image.ndmeasure import label
from skimage.measure import regionprops_table
from dask_image.ndmorph import binary_closing as binary_closing_dask
from dask_image.ndmorph import binary_opening as binary_opening_dask
from scipy.ndimage import binary_closing, binary_opening
from scipy.sparse import coo_matrix, csr_matrix, eye
from scipy.sparse.csgraph import connected_components
from dask import persist
import dask.array as dsa
from dask.base import is_dask_collection
from numba import jit, njit, prange
import warnings
import logging
import os
import shutil
import gc
import time

try:
    import jax.numpy as jnp
    HAS_JAX = True
except ImportError:
    import numpy as np
    jnp = np  # Alias for jnp
    HAS_JAX = False
    
    warnings.warn(
        "JAX not installed. Some functionality will be slower. "
        "For best performance, install with: pip install marEx[jax]",
        ImportWarning,
        stacklevel=2
    )


# ============================
# Main Tracker Class
# ============================

class tracker:
    """
    Tracker identifies and tracks arbitrary binary objects in spatial data through time.
    
    The tracker supports both structured (regular grid) and unstructured data,
    and seamlessly handles splitting & merging of objects. It identifies
    connected regions in binary data at each time step, and tracks these as 
    evolving events through time.
    
    Main workflow:
    1. Preprocessing: Fill spatiotemporal holes, filter small objects
    2. Object identification: Label connected components at each time
    3. Tracking: Determine object correspondences across time
    4. Optional splitting & merging: Handle complex event evolution
    
    Parameters
    ----------
    data_bin : xarray.DataArray
        Binary data to identify and track objects in (True = object, False = background)
    mask : xarray.DataArray
        Binary mask indicating valid regions (True = valid, False = invalid)
    R_fill : int
        Radius for filling holes/gaps in spatial domain (in grid cells)
    area_filter_quartile : float
        Quantile (0-1) for filtering smallest objects (e.g., 0.25 removes smallest 25%)
    temp_dir : str, optional
        Path to temporary directory for storing intermediate results
    T_fill : int, default=2
        Number of timesteps for filling temporal gaps (must be even)
    allow_merging : bool, default=True
        Allow objects to split and merge across time
    nn_partitioning : bool, default=False
        Use nearest-neighbor partitioning for merging events
    overlap_threshold : float, default=0.5
        Minimum fraction of overlap between objects to consider them the same
    unstructured_grid : bool, default=False
        Whether data is on an unstructured grid
    timedim : str, default='time'
        Name of time dimension
    xdim : str, default='lon'
        Name of x/longitude dimension
    ydim : str, default='lat'
        Name of y/latitude dimension
    neighbours : xarray.DataArray, optional
        For unstructured grid, indicates connectivity between cells
    cell_areas : xarray.DataArray, optional
        For unstructured grid, area of each cell
    max_iteration : int, default=40
        Maximum number of iterations for merging/splitting algorithm
    checkpoint : str, default='None'
        Checkpoint strategy ('save', 'load', or None)
    debug : int, default=0
        Debug level (0-2)
    verbosity : int, default=0
        Verbosity level
    """
        
    def __init__(self, data_bin, mask, R_fill, area_filter_quartile, 
                 temp_dir=None, T_fill=2, allow_merging=True, nn_partitioning=False, 
                 overlap_threshold=0.5, unstructured_grid=False, timedim='time', 
                 xdim='lon', ydim='lat', neighbours=None, cell_areas=None, 
                 max_iteration=40, checkpoint=None, debug=0, verbosity=0):
        
        self.data_bin = data_bin
        self.mask = mask
        self.R_fill = int(R_fill)
        self.T_fill = T_fill
        self.area_filter_quartile = area_filter_quartile
        self.allow_merging = allow_merging
        self.nn_partitioning = nn_partitioning
        self.overlap_threshold = overlap_threshold
        self.timedim = timedim
        self.xdim = xdim
        self.ydim = ydim
        self.lat = data_bin.lat.persist()
        self.lon = data_bin.lon.persist()
        self.timechunks = data_bin.chunks[data_bin.dims.index(timedim)][0]
        self.unstructured_grid = unstructured_grid
        self.mean_cell_area = 1.0  # For structured grids, units are pixels
        self.checkpoint = checkpoint
        self.debug = debug
        self.verbosity = verbosity
        
        # Input validation and preparation
        self._validate_inputs()
        
        # Special setup for unstructured grids
        if unstructured_grid:
            self._setup_unstructured_grid(temp_dir, neighbours, cell_areas, max_iteration)
        
        self._configure_warnings()
        
    def _validate_inputs(self):
        """Validate input parameters and data."""
        # For unstructured grids, adjust dimensions
        if self.unstructured_grid:
            self.ydim = None
            if ((self.timedim, self.xdim) != self.data_bin.dims):
                try:
                    self.data_bin = self.data_bin.transpose(self.timedim, self.xdim) 
                except:
                    raise ValueError(
                        f'Unstructured MarEx only supports 2D DataArrays with dimensions '
                        f'({self.timedim} and {self.xdim}). Found {list(self.data_bin.dims)}'
                    )
        else:
            # For structured grids, ensure 3D data
            if ((self.timedim, self.ydim, self.xdim) != self.data_bin.dims):
                try:
                    self.data_bin = self.data_bin.transpose(self.timedim, self.ydim, self.xdim) 
                except:
                    raise ValueError(
                        f'Gridded (structured) MarEx only supports 3D DataArrays with dimensions '
                        f'({self.timedim}, {self.ydim}, and {self.xdim}). Found {list(self.data_bin.dims)}'
                    )
        
        # Check data type and structure
        if (self.data_bin.data.dtype != bool):
            raise ValueError('The input DataArray must be binary (boolean type)')
        
        if not is_dask_collection(self.data_bin.data):
            raise ValueError('The input DataArray must be backed by a Dask array')
        
        if (self.mask.data.dtype != bool):
            raise ValueError('The mask must be binary (boolean type)')
        
        if (self.mask == False).all():
            raise ValueError('Mask contains only False values. It should indicate valid regions with True')
        
        if (self.area_filter_quartile < 0) or (self.area_filter_quartile > 1):
            raise ValueError('area_filter_quartile must be between 0 and 1')
        
        if (self.T_fill % 2 != 0):
            raise ValueError('T_fill must be even for symmetry')
        
        # Check geographic coordinates
        if ((self.lon.max().compute().item() - self.lon.min().compute().item()) < 100):
            raise ValueError('Lat/Lon coordinates must be in degrees')
    
    def _setup_unstructured_grid(self, temp_dir, neighbours, cell_areas, max_iteration):
        """Set up special handling for unstructured grids."""
        if not temp_dir:
            raise ValueError('Unstructured grid requires a temporary directory for memory-efficient processing')
        
        self.scratch_dir = temp_dir
        
        # Clear any existing temporary storage
        if os.path.exists(f'{self.scratch_dir}/marEx_temp_field.zarr/'):
            shutil.rmtree(f'{self.scratch_dir}/marEx_temp_field.zarr/')
        
        # Remove coordinate variables to avoid memory issues
        self.data_bin = self.data_bin.drop_vars({'lat', 'lon'})
        self.mask = self.mask.drop_vars({'lat', 'lon'})
        self.lat = self.lat.drop_vars(self.lat.coords)
        self.lon = self.lon.drop_vars(self.lon.coords)
        neighbours = neighbours.drop_vars({'lat', 'lon', 'nv'})
        
        self.max_iteration = max_iteration
        
        # Store cell areas (in square metres)
        self.cell_area = cell_areas.astype(np.float32).drop_vars({'lat', 'lon'}).persist()
        self.mean_cell_area = cell_areas.mean().compute().item()
        
        # Initialise dilation array for unstructured grid
        self.neighbours_int = neighbours.astype(np.int32) - 1  # Convert to 0-based indexing
        
        # Validate neighbour array structure
        if self.neighbours_int.shape[0] != 3:
            raise ValueError('Unstructured MarEx only supports triangular grids. Neighbours array must have shape (3, ncells)')
        if self.neighbours_int.dims != ('nv', self.xdim):
            raise ValueError('Neighbours array must have dimensions (nv, xdim)')
        
        # Construct sparse dilation matrix
        self._build_sparse_dilation_matrix()
    
    def _build_sparse_dilation_matrix(self):
        """Build sparse matrix for efficient dilation operations on unstructured grid."""
        # Create row and column indices for sparse matrix
        row_indices = jnp.repeat(jnp.arange(self.neighbours_int.shape[1]), 3)
        col_indices = self.neighbours_int.data.compute().T.flatten()

        # Filter out negative values (invalid connections)
        valid_mask = col_indices >= 0
        row_indices = row_indices[valid_mask]
        col_indices = col_indices[valid_mask]
        
        max_neighbour = self.neighbours_int.max().compute().item() + 1

        # Create the sparse matrix for dilation
        dilate_coo = coo_matrix(
            (jnp.ones_like(row_indices, dtype=bool), (row_indices, col_indices)), 
            shape=(self.neighbours_int.shape[1], max_neighbour)
        )
        self.dilate_sparse = csr_matrix(dilate_coo)
        
        # Add identity matrix to include self-connections
        identity = eye(self.neighbours_int.shape[1], dtype=bool, format='csr')
        self.dilate_sparse = self.dilate_sparse + identity
        
        if self.verbosity > 0:
            print('Finished constructing the sparse dilation matrix')
    
    def _configure_warnings(self):
        """Configure warning and logging suppression based on debug level."""
        if self.debug < 2:
            # Configure logging warning filters
            logging.getLogger('distributed.scheduler').setLevel(logging.ERROR)
            
            def filter_dask_warnings(record):
                msg = str(record.msg)
                
                if self.debug == 0:
                    # Suppress both run_spec and large graph warnings
                    if any(pattern in msg for pattern in [
                        'Detected different `run_spec`',
                        'Sending large graph',
                        'This may cause some slowdown'
                    ]):
                        return False
                    return True
                else:
                    # Suppress only run_spec warnings
                    if 'Detected different `run_spec`' in msg:
                        return False
                    return True

            logging.getLogger('distributed.scheduler').addFilter(filter_dask_warnings)
            
            # Configure Python warnings
            if self.debug == 0:
                warnings.filterwarnings('ignore', 
                                    category=UserWarning,
                                    module='distributed.client')
                warnings.filterwarnings('ignore', 
                                    message='.*Sending large graph.*\n.*This may cause some slowdown.*',
                                    category=UserWarning)
    
    # ============================
    # Main Public Methods
    # ============================
    
    def run(self, return_merges=False, checkpoint=None):
        """
        Run the complete object identification and tracking pipeline.
        
        This method executes the full workflow:
        1. Preprocessing: morphological operations and size filtering
        2. Identification and tracking of objects through time
        3. Computing and attaching statistics to the results
        
        Parameters
        ----------
        return_merges : bool, default=False
            If True, return merge events dataset alongside the main events
        checkpoint : str, optional
            Override the instance checkpoint setting
            
        Returns
        -------
        events_ds : xarray.Dataset
            Dataset containing tracked events and their properties
        merges_ds : xarray.Dataset, optional
            Dataset with merge event information (only if return_merges=True)
        """
        # Preprocess the binary data 
        data_bin_preprocessed, object_stats = self.run_preprocess(checkpoint=checkpoint)
        
        # Run identification and tracking
        events_ds, merges_ds, N_events_final = self.run_tracking(data_bin_preprocessed)
        
        # Compute statistics and finalise output
        events_ds = self.run_stats_attributes(events_ds, merges_ds, object_stats, N_events_final)
        
        if self.allow_merging and return_merges:
            return events_ds, merges_ds
        else:
            return events_ds
    
    def run_preprocess(self, checkpoint=None):
        """
        Preprocess binary data to prepare for tracking.
        
        This performs morphological operations to fill holes/gaps in both space and time,
        then filters small objects according to the area_filter_quartile.
        
        Parameters
        ----------
        checkpoint : str, optional
            Checkpoint strategy override
            
        Returns
        -------
        data_bin_filtered : xarray.DataArray
            Preprocessed binary data
        object_stats : tuple
            Statistics about the preprocessing
        """
        if not checkpoint:
            checkpoint = self.checkpoint
        
        def load_data_from_checkpoint():
            """Load preprocessed data from checkpoint files."""
            data_bin_preprocessed = xr.open_zarr(
                f'{self.scratch_dir}/marEx_checkpoint_proc_bin.zarr', 
                chunks={self.timedim: self.timechunks}
            )['data_bin_preproc']
            return data_bin_preprocessed
        def load_stats_from_checkpoint():
            object_stats_npz = np.load(f'{self.scratch_dir}/marEx_checkpoint_stats.npz')
            object_stats = [
                object_stats_npz[key] for key in [
                    'total_area_IDed', 'N_objects_prefiltered', 'N_objects_filtered', 
                    'area_threshold', 'accepted_area_fraction', 'preprocessed_area_fraction'
                ]
            ]
            return object_stats
        
        if checkpoint == 'load':
            print('Loading preprocessed data & stats...')
            return load_data_from_checkpoint(), load_stats_from_checkpoint()
        
        
        # Compute area of initial binary data
        raw_area = self.compute_area(self.data_bin)
        
        # Fill small holes & gaps between objects
        data_bin_filled = self.fill_holes(self.data_bin)
        del self.data_bin  # Free memory
        if self.verbosity > 0:
            print('Finished filling spatial holes')

        # Fill small time-gaps between objects
        data_bin_filled = self.fill_time_gaps(data_bin_filled).persist()
        if self.verbosity > 0:
            print('Finished filling spatio-temporal holes')
        
        # Remove small objects
        data_bin_filtered, area_threshold, object_areas, N_objects_prefiltered, N_objects_filtered = (
            self.filter_small_objects(data_bin_filled)
        )
        del data_bin_filled  # Free memory
        if self.verbosity > 0:
            print('Finished filtering small objects')
        
        # Persist preprocessed data &/or Save checkpoint
        if checkpoint and 'save' in checkpoint:
            print('Saving preprocessed data & stats...')
            time.sleep(5)
            data_bin_filtered.name = 'data_bin_preproc'
            data_bin_filtered.to_zarr(f'{self.scratch_dir}/marEx_checkpoint_proc_bin.zarr', mode='w') # N.B.: This needs to be done without .persist() due to dask to_zarr tuple bug...
            data_bin_filtered = load_data_from_checkpoint()
        else:
            data_bin_filtered = data_bin_filtered.persist()
            wait(data_bin_filtered)
                
        # Compute area of processed data
        processed_area = self.compute_area(data_bin_filtered)
        
        # Compute statistics
        object_areas = object_areas.compute()
        total_area_IDed = object_areas.sum().item()

        accepted_area = object_areas.where(object_areas > area_threshold, drop=True).sum().item()
        accepted_area_fraction = accepted_area / total_area_IDed
        
        total_hobday_area = raw_area.sum().compute().item()
        total_processed_area = processed_area.sum().compute().item()
        preprocessed_area_fraction = total_hobday_area / total_processed_area
        
        object_stats = (
            total_area_IDed, N_objects_prefiltered, N_objects_filtered, 
            area_threshold, accepted_area_fraction, preprocessed_area_fraction
        )
        
        # Save checkpoint
        if checkpoint and 'save' in checkpoint:
            np.savez(
                f'{self.scratch_dir}/marEx_checkpoint_stats.npz', 
                total_area_IDed=total_area_IDed, 
                N_objects_prefiltered=N_objects_prefiltered, 
                N_objects_filtered=N_objects_filtered, 
                area_threshold=area_threshold, 
                accepted_area_fraction=accepted_area_fraction, 
                preprocessed_area_fraction=preprocessed_area_fraction
            )
            # Reload to refresh the dask graph
            data_bin_filtered = load_data_from_checkpoint()
            object_stats = load_stats_from_checkpoint()
        
        return data_bin_filtered, object_stats
    
    def run_tracking(self, data_bin_preprocessed):
        """
        Track objects through time to identify events.
        
        Parameters
        ----------
        data_bin_preprocessed : xarray.DataArray
            Preprocessed binary data
            
        Returns
        -------
        events_ds : xarray.Dataset
            Dataset containing tracked events
        merges_ds : xarray.Dataset
            Dataset with merge information
        N_events_final : int
            Final number of unique events
        """
        if self.allow_merging or self.unstructured_grid:
            # Track with merging & splitting
            events_ds, merges_ds, N_events_final = self.track_objects(data_bin_preprocessed)
        else: 
            # Track without merging or splitting
            events_da, _, N_events_final = self.identify_objects(
                data_bin_preprocessed, time_connectivity=True
            )
            events_ds = xr.Dataset({'ID_field': events_da})
            merges_ds = xr.Dataset()
        
        # Set all filler IDs < 0 to 0
        events_ds['ID_field'] = events_ds.ID_field.where(events_ds.ID_field > 0, drop=False, other=0)
        
        # Delete the last ID -- it is all 0s
        events_ds = events_ds.isel(ID=slice(None,-1)) # At least for the gridded algorithm
        
        if self.verbosity > 0:
            print('Finished tracking all extreme events!\n\n')
        
        return events_ds, merges_ds, N_events_final
    
    def run_stats_attributes(self, events_ds, merges_ds, object_stats, N_events_final):
        """
        Add statistics and attributes to the events dataset.
        
        Parameters
        ----------
        events_ds : xarray.Dataset
            Dataset containing tracked events
        merges_ds : xarray.Dataset
            Dataset with merge information
        object_stats : tuple
            Preprocessed object statistics
        N_events_final : int
            Final number of events
            
        Returns
        -------
        events_ds : xarray.Dataset
            Dataset with added statistics and attributes
        """
        # Unpack object stats
        (total_area_IDed, N_objects_prefiltered, N_objects_filtered, 
         area_threshold, accepted_area_fraction, preprocessed_area_fraction) = object_stats

        # Add general attributes to dataset
        events_ds.attrs['allow_merging'] = int(self.allow_merging)
        events_ds.attrs['N_objects_prefiltered'] = int(N_objects_prefiltered)
        events_ds.attrs['N_objects_filtered'] = int(N_objects_filtered)
        events_ds.attrs['N_events_final'] = int(N_events_final)
        events_ds.attrs['R_fill'] = self.R_fill
        events_ds.attrs['T_fill'] = self.T_fill
        events_ds.attrs['area_filter_quartile'] = self.area_filter_quartile
        events_ds.attrs['area_threshold (cells)'] = area_threshold
        events_ds.attrs['accepted_area_fraction'] = accepted_area_fraction
        events_ds.attrs['preprocessed_area_fraction'] = preprocessed_area_fraction
        
        # Print summary statistics
        print('Tracking Statistics:')
        print(f'   Binary Hobday to Processed Area Fraction: {preprocessed_area_fraction}')
        print(f'   Total Object Area IDed (cells): {total_area_IDed}')
        print(f'   Number of Initial Pre-Filtered Objects: {N_objects_prefiltered}')
        print(f'   Number of Final Filtered Objects: {N_objects_filtered}')
        print(f'   Area Cutoff Threshold (cells): {area_threshold.astype(np.int32)}')
        print(f'   Accepted Area Fraction: {accepted_area_fraction}')
        print(f'   Total Events Tracked: {N_events_final}')
        
        # Add merge-specific attributes if applicable
        if self.allow_merging:
            events_ds.attrs['overlap_threshold'] = self.overlap_threshold
            events_ds.attrs['nn_partitioning'] = int(self.nn_partitioning)
            
            # Add merge summary attributes 
            events_ds.attrs['total_merges'] = len(merges_ds.merge_ID)
            events_ds.attrs['multi_parent_merges'] = (merges_ds.n_parents > 2).sum().item()
            
            print(f"   Total Merging Events Recorded: {events_ds.attrs['total_merges']}")
        
        # For unstructured grid, restore coordinates and rechunk
        if self.unstructured_grid:
            # Add lat & lon back as coordinates
            events_ds = events_ds.assign_coords(
                lat=self.lat.compute(), 
                lon=self.lon.compute()
            )
            # Rechunk to size 1 for better post-processing
            events_ds = events_ds.chunk({self.timedim: 1})
        
        return events_ds
    
    # ============================
    # Data Processing Methods
    # ============================
    
    def compute_area(self, data_bin):
        """
        Compute the total area of binary data at each time.
        
        Parameters
        ----------
        data_bin : xarray.DataArray
            Binary data
            
        Returns
        -------
        area : xarray.DataArray
            Total area at each time (units: pixels for structured grid, matching cell_area for unstructured)
        """
        if self.unstructured_grid:
            area = (data_bin * self.cell_area).sum(dim=[self.xdim])
        else:
            area = data_bin.sum(dim=[self.ydim, self.xdim])
        
        return area
    
    def fill_holes(self, data_bin, R_fill=None):
        """
        Fill holes and gaps using morphological operations.
        
        This performs closing (dilation followed by erosion) to fill small gaps,
        then opening (erosion followed by dilation) to remove small isolated objects.
        
        Parameters
        ----------
        data_bin : xarray.DataArray
            Binary data to process
        R_fill : int, optional
            Fill radius override
            
        Returns
        -------
        data_bin_filled : xarray.DataArray
            Binary data with holes/gaps filled
        """
        if R_fill is None:
            R_fill = self.R_fill
        
        if self.unstructured_grid:
            # Process unstructured grid using sparse matrix operations
            # _Put the data into an xarray.DataArray to pass into the apply_ufunc_ -- Needed for correct memory management !
            sp_data = xr.DataArray(self.dilate_sparse.data, dims='sp_data')
            indices = xr.DataArray(self.dilate_sparse.indices, dims='indices')
            indptr = xr.DataArray(self.dilate_sparse.indptr, dims='indptr')
            
            def binary_open_close(bitmap_binary, sp_data, indices, indptr, mask):
                """
                Binary opening and closing for unstructured grid.
                Uses sparse matrix power operations for efficiency.
                """
                ## Closing: Dilation then Erosion (fills small gaps)
                
                # Dilation
                bitmap_binary = sparse_bool_power(bitmap_binary, sp_data, indices, indptr, R_fill)
                
                # Set land values to True (to avoid artificially eroding the shore)
                bitmap_binary[:, ~mask] = True
                
                # Erosion (negated dilation of negated image)
                bitmap_binary = ~sparse_bool_power(~bitmap_binary, sp_data, indices, indptr, R_fill)
                
                ## Opening: Erosion then Dilation (removes small objects)
                
                # Set land values to True (to avoid artificially eroding the shore)
                bitmap_binary[:, ~mask] = True
                
                # Erosion
                bitmap_binary = ~sparse_bool_power(~bitmap_binary, sp_data, indices, indptr, R_fill)
                
                # Dilation
                bitmap_binary = sparse_bool_power(bitmap_binary, sp_data, indices, indptr, R_fill)
                
                return bitmap_binary
            
            # Apply the operations
            data_bin = xr.apply_ufunc(
                binary_open_close, data_bin, sp_data, indices, indptr, self.mask, 
                input_core_dims=[[self.xdim], ['sp_data'], ['indices'], ['indptr'], [self.xdim]],
                output_core_dims=[[self.xdim]],
                output_dtypes=[np.bool_],
                vectorize=False,
                dask_gufunc_kwargs={'output_sizes': {self.xdim: data_bin.sizes[self.xdim]}},
                dask='parallelized'
            )
        
        else:
            # Structured grid using dask-powered morphological operations
            use_dask_morph = True
            
            # Generate structuring element (disk-shaped)
            y, x = np.ogrid[-R_fill:R_fill+1, -R_fill:R_fill+1]
            r = x**2 + y**2
            diameter = 2 * R_fill
            se_kernel = r < (R_fill**2)+1
            
            if use_dask_morph:
                # Pad data to avoid edge effects
                data_bin = data_bin.pad({self.ydim: diameter, self.xdim: diameter}, mode='wrap')
                data_coords = data_bin.coords
                data_dims = data_bin.dims
                
                # Apply morphological operations
                data_bin = binary_closing_dask(data_bin.data, structure=se_kernel[np.newaxis, :, :]) # N.B.: There may be a rearing bug in constructing the dask task graph when we extract and then re-imbed the dask array into an xarray DataArray
                data_bin = binary_opening_dask(data_bin, structure=se_kernel[np.newaxis, :, :])
                
                # Convert back to xarray.DataArray and trim padding
                data_bin = xr.DataArray(data_bin, coords=data_coords, dims=data_dims)
                data_bin = data_bin.isel({
                    self.ydim: slice(diameter, -diameter), 
                    self.xdim: slice(diameter, -diameter)
                })
            else:
                def binary_open_close(bitmap_binary):
                    """Apply binary opening and closing in one function."""
                    bitmap_binary_padded = np.pad(
                        bitmap_binary,
                        ((diameter, diameter), (diameter, diameter)),
                        mode='wrap'
                    )
                    s1 = binary_closing(bitmap_binary_padded, se_kernel, iterations=1)
                    s2 = binary_opening(s1, se_kernel, iterations=1)
                    unpadded = s2[diameter:-diameter, diameter:-diameter]
                    return unpadded

                data_bin = xr.apply_ufunc(
                    binary_open_close, data_bin,
                    input_core_dims=[[self.ydim, self.xdim]],
                    output_core_dims=[[self.ydim, self.xdim]],
                    output_dtypes=[data_bin.dtype],
                    vectorize=True,
                    dask='parallelized'
                )
            
            # Mask out edge features from morphological operations
            data_bin = data_bin.where(self.mask, drop=False, other=False)
        
        return data_bin
    
    def fill_time_gaps(self, data_bin):
        """
        Fill temporal gaps between objects.
        
        Performs binary closing (dilation then erosion) along the time dimension 
        to fill small time gaps between objects.
        
        Parameters
        ----------
        data_bin : xarray.DataArray
            Binary data to process
            
        Returns
        -------
        data_bin_filled : xarray.DataArray
            Binary data with temporal gaps filled
        """
        if self.T_fill == 0:
            return data_bin
        
        # Create temporal structuring element
        kernel_size = self.T_fill + 1  # This will then fill a maximum hole size of self.T_fill
        time_kernel = np.ones(kernel_size, dtype=bool)
        
        if self.ydim is None:
            # Unstructured grid has only 1 additional dimension
            time_kernel = time_kernel[:, np.newaxis]
        else: 
            time_kernel = time_kernel[:, np.newaxis, np.newaxis]
        
        # Pad in time to avoid edge effects
        data_bin = data_bin.pad({self.timedim: kernel_size}, mode='constant', constant_values=False)
        
        # Apply temporal closing
        data_bin_dask = data_bin.data
        closed_dask_array = binary_closing_dask(data_bin_dask, structure=time_kernel)
        
        # Convert back to xarray.DataArray
        data_bin_filled = xr.DataArray(
            closed_dask_array,
            coords=data_bin.coords,
            dims=data_bin.dims,
            attrs=data_bin.attrs
        )
        
        # Remove padding
        data_bin_filled = data_bin_filled.isel({self.timedim: slice(kernel_size, -kernel_size)}).persist()
        
        # Fill newly-created spatial holes
        data_bin_filled = self.fill_holes(data_bin_filled, R_fill=self.R_fill//2)
        
        return data_bin_filled
    
    def refresh_dask_graph(self, data_bin):
        """
        Clear and reset the Dask graph via save/load cycle.
        
        This is needed to work around a memory leak bug in Dask where
        "Unmanaged Memory" builds up within loops.
        
        Parameters
        ----------
        data_bin : xarray.DataArray
            Data to refresh
            
        Returns
        -------
        data_new : xarray.DataArray
            Data with fresh Dask graph
        """
        if self.verbosity > 1:
            print('  Refreshing Dask task graph...')
        
        data_bin.name = 'temp'
        data_bin.to_zarr(f'{self.scratch_dir}/marEx_temp_field.zarr', mode='w')
        del data_bin
        gc.collect()
        
        data_new = xr.open_zarr(f'{self.scratch_dir}/marEx_temp_field.zarr', chunks={}).temp
        return data_new
    
    def filter_small_objects(self, data_bin):
        """
        Remove objects smaller than a threshold area.
        
        Parameters
        ----------
        data_bin : xarray.DataArray
            Binary data to filter
            
        Returns
        -------
        data_bin_filtered : xarray.DataArray
            Binary data with small objects removed
        area_threshold : float
            Area threshold used for filtering
        object_areas : xarray.DataArray
            Areas of all objects pre-filtering
        N_objects_prefiltered : int
            Number of objects before filtering
        N_objects_filtered : int
            Number of objects after filtering
        """
        # Cluster & Label Binary Data: Time-independent in 2D (i.e. no time connectivity!)
        object_id_field, _, N_objects_unfiltered = self.identify_objects(data_bin, time_connectivity=False)
        
        if self.unstructured_grid:
            # Get the maximum ID to dimension arrays
            #  Note: identify_objects() starts at ID=0 for every time slice
            max_ID = object_id_field.max().compute().item()
            
            def count_cluster_sizes(object_id_field):
                """Count the number of cells in each cluster."""
                unique, counts = np.unique(object_id_field[object_id_field > 0], return_counts=True)
                padded_sizes = np.zeros(max_ID, dtype=np.int32)
                padded_unique = np.zeros(max_ID, dtype=np.int32)
                padded_sizes[:len(counts)] = counts
                padded_unique[:len(counts)] = unique
                return padded_sizes, padded_unique
            
            # Calculate cluster sizes
            cluster_sizes, unique_cluster_IDs = xr.apply_ufunc(
                count_cluster_sizes, 
                object_id_field, 
                input_core_dims=[[self.xdim]],
                output_core_dims=[['ID'], ['ID']],
                dask_gufunc_kwargs={'output_sizes': {'ID': max_ID}},
                output_dtypes=(np.int32, np.int32),
                vectorize=True,
                dask='parallelized'
            )
                    
            results = persist(cluster_sizes, unique_cluster_IDs)
            cluster_sizes, unique_cluster_IDs = results
            
            # Pre-filter tiny objects for performance
            cluster_sizes_filtered_dask = cluster_sizes.where(cluster_sizes > 50).data
            cluster_areas_mask = dsa.isfinite(cluster_sizes_filtered_dask)
            object_areas = cluster_sizes_filtered_dask[cluster_areas_mask].compute()
            
            # Filter based on area threshold
            N_objects_unfiltered = len(object_areas)
            area_threshold = np.percentile(object_areas, self.area_filter_quartile*100)
            N_objects_filtered = np.sum(object_areas > area_threshold)
            
            def filter_area_binary(cluster_IDs_0, keep_IDs_0):
                """Keep only clusters above threshold area."""
                keep_IDs_0 = keep_IDs_0[keep_IDs_0 > 0]
                keep_where = np.isin(cluster_IDs_0, keep_IDs_0)
                return keep_where
            
            # Create filtered binary data
            keep_IDs = xr.where(cluster_sizes > area_threshold, unique_cluster_IDs, 0)
            
            data_bin_filtered = xr.apply_ufunc(
                filter_area_binary, 
                object_id_field, keep_IDs, 
                input_core_dims=[[self.xdim], ['ID']],
                output_core_dims=[[self.xdim]],
                output_dtypes=[data_bin.dtype],
                vectorize=True,
                dask='parallelized'
            )
            
            object_areas = cluster_sizes  # Store pre-filtered areas
            
        else:
            # Structured grid approach
            
            # Calculate object properties including area
            object_props = self.calculate_object_properties(object_id_field)
            object_areas, object_ids = object_props.area, object_props.ID
            
            # Calculate area threshold
            area_threshold = np.percentile(object_areas, self.area_filter_quartile*100.0)
            
            # Keep only objects above threshold
            object_ids_keep = xr.where(object_areas >= area_threshold, object_ids, -1)
            object_ids_keep[0] = -1  # Don't keep ID=0
            
            # Create filtered binary data
            data_bin_filtered = object_id_field.isin(object_ids_keep)
            
            # Count objects after filtering
            N_objects_filtered = object_ids_keep.where(object_ids_keep > 0).count().item()
        
        return data_bin_filtered, area_threshold, object_areas, N_objects_unfiltered, N_objects_filtered
    
    # ============================
    # Object Identification Methods
    # ============================
    
    def identify_objects(self, data_bin, time_connectivity):
        """
        Identify connected regions in binary data.
        
        Parameters
        ----------
        data_bin : xarray.DataArray
            Binary data to identify objects in
        time_connectivity : bool
            Whether to connect objects across time
            
        Returns
        -------
        object_id_field : xarray.DataArray
            Field of integer IDs for each object
        None : NoneType
            Placeholder for compatibility with track_objects
        N_objects : int
            Number of objects identified
        """
        if self.unstructured_grid:
            # The resulting ID field for unstructured grid will start at 0 for each time-slice,
            # which differs from structured grid where IDs are unique across time.
            
            if time_connectivity:
                raise ValueError('Cannot automatically compute time-connectivity on unstructured grid')
            
            # Use Union-Find (Disjoint Set Union) clustering for unstructured grid
            def cluster_true_values(arr, neighbours_int):
                """Cluster connected True values in binary data on unstructured grid."""
                t, n = arr.shape
                labels = np.full((t, n), -1, dtype=np.int32)
                
                for i in range(t):
                    # Get indices of True values
                    true_indices = np.where(arr[i])[0]
                    mapping = {old_idx: new_idx for new_idx, old_idx in enumerate(true_indices)}
                    
                    # Find connected components
                    valid_mask = (neighbours_int != -1) & arr[i][neighbours_int]
                    row_ind, col_ind = np.where(valid_mask)
                    
                    # Map to compact indices for graph algorithm
                    mapped_row_ind = []
                    mapped_col_ind = []
                    for r, c in zip(neighbours_int[row_ind, col_ind], col_ind):
                        if r in mapping and c in mapping:
                            mapped_row_ind.append(mapping[r])
                            mapped_col_ind.append(mapping[c])
                    
                    # Create graph and find connected components
                    graph = csr_matrix(
                        (np.ones(len(mapped_row_ind)), (mapped_row_ind, mapped_col_ind)), 
                        shape=(len(true_indices), len(true_indices))
                    )
                    _, labels_true = connected_components(csgraph=graph, directed=False, return_labels=True)
                    labels[i, true_indices] = labels_true
                
                return labels + 1  # Add 1 so 0 represents no object
            
            # Apply mask and cluster
            data_bin = data_bin.where(self.mask, other=False)
            
            object_id_field = xr.apply_ufunc(
                cluster_true_values, 
                data_bin, 
                self.neighbours_int, 
                input_core_dims=[[self.xdim], ['nv', self.xdim]],
                output_core_dims=[[self.xdim]],
                output_dtypes=[np.int32],
                dask_gufunc_kwargs={'output_sizes': {self.xdim: data_bin.sizes[self.xdim]}},
                vectorize=False,
                dask='parallelized'
            )
            
            # Ensure ID = 0 on invalid regions
            object_id_field = object_id_field.where(self.mask, other=0)
            object_id_field = object_id_field.persist()
            object_id_field = object_id_field.rename('ID_field')
            N_objects = 1  # Placeholder (IDs aren't unique across time)
            
        else:  # Structured Grid
            # Create connectivity kernel for labeling
            neighbours = np.zeros((3, 3, 3))
            
            if time_connectivity:
                # ID objects in 3D (i.e. space & time) -- N.B. IDs are unique across time
                neighbours[:, :, :] = 1 #                       including +-1 in time, _and also diagonal in time_ -- i.e. edges can touch
            else:
                # ID objects only in 2D (i.e. space) -- N.B. IDs are _not_ unique across time (i.e. each time starts at 0 again)   
                neighbours[1, :, :] = 1 # All 8 neighbours, but ignore time
            
            # Cluster & label binary data
            object_id_field, N_objects = label(           # Apply dask-powered ndimage & persist in memory
                data_bin,
                structure=neighbours, 
                wrap_axes=(2,)  # Wrap in x-direction
            )
            results = persist(object_id_field, N_objects)
            object_id_field, N_objects = results
            
            N_objects = N_objects.compute()
            
            # Convert to DataArray with same coordinates as input
            object_id_field = xr.DataArray(
                object_id_field, 
                coords=data_bin.coords, 
                dims=data_bin.dims, 
                attrs=data_bin.attrs
            ).rename('ID_field').astype(np.int32)
        
        return object_id_field, None, N_objects
    
    def calculate_centroid(self, binary_mask, original_centroid=None):
        """
        Calculate object centroid, handling edge cases for periodic boundaries.
        
        Parameters
        ----------
        binary_mask : numpy.ndarray
            2D binary array where True indicates the object (dimensions are (y,x))
        original_centroid : tuple, optional
            (y_centroid, x_centroid) from regionprops_table
            
        Returns
        -------
        tuple
            (y_centroid, x_centroid)
        """
        # Check if object is near either edge of x dimension
        near_left_BC = np.any(binary_mask[:, :100])
        near_right_BC = np.any(binary_mask[:, -100:])
        
        if original_centroid is None:
            # Calculate y centroid from scratch
            y_indices = np.nonzero(binary_mask)[0]
            y_centroid = np.mean(y_indices)
        else: 
            y_centroid = original_centroid[0]
        
        # If object is near both edges, recalculate x-centroid to handle wrapping
        # N.B.: We calculate _near_ rather than touching, to catch the edge case where the object may be split and straddling the boundary !
        if near_left_BC and near_right_BC:
            # Adjust x coordinates that are near right edge
            x_indices = np.nonzero(binary_mask)[1]
            x_indices_adj = x_indices.copy()
            right_side = x_indices > binary_mask.shape[1] // 2
            x_indices_adj[right_side] -= binary_mask.shape[1]
            
            x_centroid = np.mean(x_indices_adj)
            if x_centroid < 0:  # Ensure centroid is positive
                x_centroid += binary_mask.shape[1]
                
        elif original_centroid is None:
            # Calculate x-centroid from scratch
            x_indices = np.nonzero(binary_mask)[1]
            x_centroid = np.mean(x_indices)
            
        else: 
            x_centroid = original_centroid[1]
        
        return (y_centroid, x_centroid)
    
    def calculate_object_properties(self, object_id_field, properties=None):
        """
        Calculate properties of objects from ID field.
        
        Parameters
        ----------
        object_id_field : xarray.DataArray
            Field containing object IDs
        properties : list, optional
            List of properties to calculate (defaults to ['label', 'area'])
            
        Returns
        -------
        object_props : xarray.Dataset
            Dataset containing calculated properties with 'ID' dimension
        """
        # Set default properties
        if properties is None:
            properties = ['label', 'area']
        
        # Ensure 'label' is included
        if 'label' not in properties:
            properties = ['label'] + properties  # 'label' is actually 'ID' within regionprops
        
        check_centroids = 'centroid' in properties
        
        if self.unstructured_grid:
            # Compute properties on unstructured grid
            
            # Convert lat/lon to radians
            lat_rad = np.radians(self.lat)
            lon_rad = np.radians(self.lon)
            
            # Calculate buffer size for IDs in chunks
            max_ID = object_id_field.max().compute().item() + 1
            ID_buffer_size = int(max_ID / object_id_field[self.timedim].shape[0]) * 4
            
            def object_properties_chunk(ids, lat, lon, area, buffer_IDs=True):
                """
                Calculate object properties for a chunk of data.
                Uses vectorised operations for efficiency.
                """
                # Find valid IDs
                valid_mask = ids > 0
                ids_chunk = np.unique(ids[valid_mask])
                n_ids = len(ids_chunk)
                
                if n_ids == 0:
                    # No objects in this chunk
                    if buffer_IDs:
                        result = np.zeros((3, ID_buffer_size), dtype=np.float32)
                        padded_ids = np.zeros(ID_buffer_size, dtype=np.int32)
                        return result, padded_ids
                    else:
                        result = np.zeros((3, 0), dtype=np.float32)
                        padded_ids = np.array([], dtype=np.int32)
                        return result, padded_ids
                
                # Map IDs to consecutive indices
                mapped_indices = np.searchsorted(ids_chunk, ids[valid_mask])
                
                # Pre-allocate arrays
                areas = np.zeros(n_ids, dtype=np.float32)
                weighted_x = np.zeros(n_ids, dtype=np.float32)
                weighted_y = np.zeros(n_ids, dtype=np.float32)
                weighted_z = np.zeros(n_ids, dtype=np.float32)
                
                # Convert to Cartesian for centroid calculation
                cos_lat = np.cos(lat[valid_mask])
                x = cos_lat * np.cos(lon[valid_mask])
                y = cos_lat * np.sin(lon[valid_mask])
                z = np.sin(lat[valid_mask])
                
                # Compute areas
                valid_areas = area[valid_mask]
                np.add.at(areas, mapped_indices, valid_areas)
                
                # Compute weighted coordinates
                np.add.at(weighted_x, mapped_indices, valid_areas * x)
                np.add.at(weighted_y, mapped_indices, valid_areas * y)
                np.add.at(weighted_z, mapped_indices, valid_areas * z)
                
                # Clean intermediate arrays
                del x, y, z, cos_lat, valid_areas
                
                # Normalise vectors
                norm = np.sqrt(weighted_x**2 + weighted_y**2 + weighted_z**2)
                norm = np.where(norm > 0, norm, 1)  # Avoid division by zero
                
                weighted_x /= norm
                weighted_y /= norm
                weighted_z /= norm
                
                # Convert back to lat/lon
                centroid_lat = np.degrees(np.arcsin(np.clip(weighted_z, -1, 1)))
                centroid_lon = np.degrees(np.arctan2(weighted_y, weighted_x))
                
                # Fix longitude range to [-180, 180]
                centroid_lon = np.where(
                    centroid_lon > 180., centroid_lon - 360.,
                    np.where(centroid_lon < -180., centroid_lon + 360., centroid_lon)
                )
                
                if buffer_IDs:
                    # Create padded output arrays
                    result = np.zeros((3, ID_buffer_size), dtype=np.float32)
                    padded_ids = np.zeros(ID_buffer_size, dtype=np.int32)
                    
                    # Fill arrays up to n_ids
                    result[0, :n_ids] = areas
                    result[1, :n_ids] = centroid_lat
                    result[2, :n_ids] = centroid_lon
                    padded_ids[:n_ids] = ids_chunk
                else:
                    result = np.vstack((areas, centroid_lat, centroid_lon))
                    padded_ids = ids_chunk
                
                return result, padded_ids
            
            # Process single time or multiple times
            if object_id_field[self.timedim].size == 1:
                props_np, ids = object_properties_chunk(
                    object_id_field.values, 
                    lat_rad.values, 
                    lon_rad.values, 
                    self.cell_area.values, 
                    buffer_IDs=False
                )
                props = xr.DataArray(props_np, dims=['prop', 'out_id'])
            
            else:
                # Process in parallel
                props_buffer, ids_buffer = xr.apply_ufunc(
                    object_properties_chunk,
                    object_id_field,
                    lat_rad,
                    lon_rad,
                    self.cell_area,
                    input_core_dims=[[self.xdim], [self.xdim], [self.xdim], [self.xdim]],
                    output_core_dims=[['prop', 'out_id'], ['out_id']],
                    output_dtypes=[np.float32, np.int32],
                    dask_gufunc_kwargs={'output_sizes': {'prop': 3, 'out_id': ID_buffer_size}},
                    vectorize=True,
                    dask='parallelized'
                )
                results = persist(props_buffer, ids_buffer)
                props_buffer, ids_buffer = results
                ids_buffer = ids_buffer.compute().values.reshape(-1)
                
                # Get valid IDs (non-zero)
                valid_ids_mask = ids_buffer > 0
                
                # Check if we have any valid IDs before stacking
                if np.any(valid_ids_mask):
                    ids = ids_buffer[valid_ids_mask]
                    props = props_buffer.stack(combined=('time', 'out_id')).isel(combined=valid_ids_mask)
                else:
                    # No valid IDs found
                    ids = np.array([], dtype=np.int32)
                    props = xr.DataArray(np.zeros((3, 0), dtype=np.float32), dims=['prop', 'out_id'])

            # Create object properties dataset
            if len(ids) > 0:
                object_props = xr.Dataset(
                    {
                        'area': ('out_id', props.isel(prop=0).data),
                        'centroid-0': ('out_id', props.isel(prop=1).data),
                        'centroid-1': ('out_id', props.isel(prop=2).data)
                    },
                    coords={'ID': ('out_id', ids)}
                ).set_index(out_id='ID').rename({'out_id': 'ID'})
            else:
                # Create empty dataset with correct structure
                object_props = xr.Dataset(
                    {
                        'area': ('ID', []), 
                        'centroid-0': ('ID', []), 
                        'centroid-1': ('ID', [])
                    },
                    coords={'ID': []}
                )
            
        else:
            # Structured grid approach
            # N.B.: These operations are simply done on a pixel grid — no cartesian conversion (therefore, polar regions are doubly biased)
            
            # Define function to calculate properties for each chunk
            def object_properties_chunk(ids):
                """Calculate object properties for a chunk of data."""
                # Use regionprops_table for standard properties
                props_slice = regionprops_table(ids, properties=properties)
                
                # Handle centroid calculation for objects that wrap around edges
                if check_centroids and len(props_slice['label']) > 0:
                    # Get original centroids
                    centroids = list(zip(props_slice['centroid-0'], props_slice['centroid-1']))
                    centroids_wrapped = []
                    
                    # Process each object
                    for ID_idx, ID in enumerate(props_slice['label']):
                        binary_mask = ids == ID
                        centroids_wrapped.append(
                            self.calculate_centroid(binary_mask, centroids[ID_idx])
                        )
                    
                    # Update centroid values
                    props_slice['centroid-0'] = [c[0] for c in centroids_wrapped]
                    props_slice['centroid-1'] = [c[1] for c in centroids_wrapped]
                
                return props_slice
            
            # Process single time or multiple times
            if object_id_field[self.timedim].size == 1:
                object_props = object_properties_chunk(object_id_field.values)
                object_props = xr.Dataset({key: (['ID'], value) for key, value in object_props.items()})
            else:
                # Run in parallel
                object_props = xr.apply_ufunc(
                    object_properties_chunk, 
                    object_id_field,
                    input_core_dims=[[self.ydim, self.xdim]],
                    output_core_dims=[[]],
                    output_dtypes=[object],
                    vectorize=True,
                    dask='parallelized'
                )
                
                # Concatenate and convert to dataset
                object_props = xr.concat([
                    xr.Dataset({key: (['ID'], value) for key, value in item.items()}) 
                    for item in object_props.values
                ], dim='ID')
            
            # Set ID as coordinate
            object_props = object_props.set_index(ID='label')
        
        # Combine centroid components into a single variable
        if 'centroid' in properties and len(object_props.ID) > 0:
            object_props['centroid'] = xr.concat(
                [object_props['centroid-0'], object_props['centroid-1']], 
                dim='component'
            )
            object_props = object_props.drop_vars(['centroid-0', 'centroid-1'])
        
        return object_props
    
    # ============================
    # Overlap and Tracking Methods
    # ============================
    
    def check_overlap_slice(self, ids_t0, ids_next):
        """
        Find overlapping objects between two consecutive time slices.
        
        Parameters
        ----------
        ids_t0 : numpy.ndarray
            Object IDs at current time
        ids_next : numpy.ndarray
            Object IDs at next time
            
        Returns
        -------
        numpy.ndarray
            Array of shape (n_overlaps, 3) with [id_t0, id_next, overlap_area]
        """
        # Create masks for valid IDs
        mask_t0 = ids_t0 > 0
        mask_next = ids_next > 0
        
        # Only process cells where both times have valid IDs
        combined_mask = mask_t0 & mask_next
        
        if not np.any(combined_mask):
            return np.empty((0, 3), dtype=np.float32 if self.unstructured_grid else np.int32)
        
        # Extract the overlapping points
        ids_t0_valid = ids_t0[combined_mask]
        ids_next_valid = ids_next[combined_mask]
        
        # Create a unique identifier for each pair
        # This is faster than using np.unique with axis=1
        max_id = max(ids_t0.max(), ids_next.max() + 1).astype(np.int64)
        pair_ids = ids_t0_valid.astype(np.int64) * max_id + ids_next_valid.astype(np.int64)
        
        if self.unstructured_grid:
            # Get unique pairs and their inverse indices
            unique_pairs, inverse_indices = np.unique(pair_ids, return_inverse=True)

            # Sum areas for overlapping cells
            areas_valid = self.cell_area.values[combined_mask]
            areas = np.zeros(len(unique_pairs), dtype=np.float32)
            np.add.at(areas, inverse_indices, areas_valid)
        else:
            # Get unique pairs and their counts (pixel counts)
            unique_pairs, areas = np.unique(pair_ids, return_counts=True)
            areas = areas.astype(np.int32)
        
        # Convert back to original ID pairs
        id_t0 = (unique_pairs // max_id).astype(np.int32)
        id_next = (unique_pairs % max_id).astype(np.int32)
            
        # Stack results
        result = np.column_stack((id_t0, id_next, areas))
        
        return result
    
    def find_overlapping_objects(self, object_id_field):
        """
        Find all overlapping objects across time.
        
        Parameters
        ----------
        object_id_field : xarray.DataArray
            Field containing object IDs
            
        Returns
        -------
        overlap_objects_list_unique_filtered : (N x 3) numpy.ndarray
            Array of object ID pairs that overlap across time, with overlap area
            The object in the first column precedes the second column in time. 
            The third column contains:
                - For structured grid: number of overlapping pixels (int32)
                - For unstructured grid: total overlapping area in m^2 (float32)
        """
        ## Check just for overlap with next time slice.
        #  Keep a running list of all object IDs that overlap
        object_id_field_next = object_id_field.shift({self.timedim: -1}, fill_value=0)

        # Calculate overlaps in parallel
        input_dims = [self.xdim] if self.unstructured_grid else [self.ydim, self.xdim]
        overlap_object_pairs_list = xr.apply_ufunc(
            self.check_overlap_slice,
            object_id_field,
            object_id_field_next,
            input_core_dims=[input_dims, input_dims],
            output_core_dims=[[]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[object]
        ).persist()
        
        # Concatenate all pairs from different chunks
        all_pairs_with_areas = np.concatenate(overlap_object_pairs_list.values)
        
        # Get unique pairs and their indices
        unique_pairs, inverse_indices = np.unique(all_pairs_with_areas[:, :2], axis=0, return_inverse=True)

        # Sum the overlap areas using the inverse indices
        output_dtype = np.float32 if self.unstructured_grid else np.int32
        total_summed_areas = np.zeros(len(unique_pairs), dtype=output_dtype)
        np.add.at(total_summed_areas, inverse_indices, all_pairs_with_areas[:, 2])

        # Stack the pairs with their summed areas
        overlap_objects_list_unique = np.column_stack((unique_pairs, total_summed_areas))
        
        return overlap_objects_list_unique
    
    def enforce_overlap_threshold(self, overlap_objects_list, object_props):
        """
        Filter object pairs based on overlap threshold.
        
        Parameters
        ----------
        overlap_objects_list : (N x 3) numpy.ndarray
            Array of object ID pairs with overlap area
        object_props : xarray.Dataset
            Object properties including area
            
        Returns
        -------
        overlap_objects_list_filtered : (M x 3) numpy.ndarray
            Filtered array of object ID pairs that meet the overlap threshold
        """
        if len(overlap_objects_list) == 0:
            return np.empty((0, 3), dtype=np.float32 if self.unstructured_grid else np.int32)
        
        # Calculate overlap fractions
        areas_0 = object_props['area'].sel(ID=overlap_objects_list[:, 0]).values
        areas_1 = object_props['area'].sel(ID=overlap_objects_list[:, 1]).values
        min_areas = np.minimum(areas_0, areas_1)
        overlap_fractions = overlap_objects_list[:, 2].astype(float) / min_areas
        
        # Filter by threshold
        overlap_objects_list_filtered = overlap_objects_list[overlap_fractions >= self.overlap_threshold]
        
        return overlap_objects_list_filtered
    
    def compute_id_time_dict(self, da, child_objects, max_objects, all_objects=True):
        """
        Generate lookup table mapping object IDs to their time index.
        
        Parameters
        ----------
        da : xarray.DataArray
            Field of object IDs
        child_objects : list or array
            Object IDs to include in the dictionary
        max_objects : int
            Maximum number of objects
        all_objects : bool, default=True
            Whether to process all objects or just child_objects
            
        Returns
        -------
        time_index_map : dict
            Dictionary mapping object IDs to time indices
        """
        # Estimate max objects per time
        est_objects_per_time_max = int(max_objects / da[self.timedim].shape[0] * 100)

        def unique_pad(x):
            """Extract unique values and pad to fixed size."""
            uniq = np.unique(x)
            result = np.zeros(est_objects_per_time_max, dtype=x.dtype) # Pad output to maximum size
            result[:len(uniq)] = uniq
            return result

        # Get unique IDs for each time slice
        input_dims = [self.xdim] if self.unstructured_grid else [self.ydim, self.xdim]
        unique_ids_by_time = xr.apply_ufunc(
            unique_pad,
            da,
            input_core_dims=[input_dims],
            output_core_dims=[['unique_values']],
            dask='parallelized',
            vectorize=True,
            dask_gufunc_kwargs={'output_sizes': {'unique_values': est_objects_per_time_max}}
        )
        
        # Set up IDs to search for
        if not all_objects:
            # Just search for the specified child objects
            search_ids = xr.DataArray(
                child_objects,
                dims=['child_id'],
                coords={'child_id': child_objects}
            )
        else:
            # Search for all possible IDs
            search_ids = xr.DataArray(
                np.arange(max_objects, dtype=np.int32),
                dims=['child_id'],
                coords={'child_id': np.arange(max_objects)}
            ).chunk({'child_id': 10000})  # Chunk for better parallelism
            
        # Find the first time index where each ID appears
        time_indices = ((unique_ids_by_time == search_ids)
                .any(dim=['unique_values'])
                .argmax(dim=self.timedim).compute())
        
        # Convert to dictionary for fast lookup
        time_index_map = {
            int(id_val): int(idx.values) 
            for id_val, idx in zip(time_indices.child_id, time_indices)
        }
        
        return time_index_map
    
    # ============================
    # Event Tracking Methods
    # ============================
    
    def track_objects(self, data_bin):
        """
        Track objects through time to form events.
        
        This is the main tracking method that handles splitting and merging of objects.
        
        Parameters
        ----------
        data_bin : xarray.DataArray
            Preprocessed binary data:  Field of globally unique integer IDs of each element in connected regions. ID = 0 indicates no object.
            
        Returns
        -------
        split_merged_events_ds : xarray.Dataset
            Dataset containing tracked events
        merge_events : xarray.Dataset
            Dataset with merge information
        N_events : int
            Final number of events
        """
        # Identify objects at each time step
        object_id_field, _, _ = self.identify_objects(data_bin, time_connectivity=False)
        object_id_field = object_id_field.persist()
        del data_bin
        if self.verbosity > 0:
            print('Finished object identification')
        
        # For unstructured grid, make objects unique across time
        if self.unstructured_grid:
            cumsum_ids = (object_id_field.max(dim=self.xdim)).cumsum(self.timedim).shift({self.timedim: 1}, fill_value=0)
            object_id_field = xr.where(object_id_field > 0, object_id_field + cumsum_ids, 0)
            object_id_field = self.refresh_dask_graph(object_id_field)
            if self.verbosity > 0:
                print(f'Finished assigning c. {cumsum_ids.max().compute().values} globally unique object IDs')
        
        # Calculate object properties
        object_props = self.calculate_object_properties(object_id_field, properties=['area', 'centroid'])
        object_props = object_props.persist()
        wait(object_props)
        if self.verbosity > 0:
            print('Finished calculating object properties')
        
        # Apply splitting & merging logic
        #  This is the most intricate step due to non-trivial loop-wise dependencies
        #  In v2.0_unstruct, this loop has been painstakingly parallelised
        split_and_merge = self.split_and_merge_objects_parallel if self.unstructured_grid else self.split_and_merge_objects
        object_id_field, object_props, overlap_objects_list, merge_events = split_and_merge(object_id_field, object_props)
        if self.verbosity > 0:
            print('Finished splitting and merging objects')
        
        # Persist results (This helps avoid block-wise task fusion run_spec issues with dask)
        results = persist(object_id_field, object_props, overlap_objects_list, merge_events)
        object_id_field, object_props, overlap_objects_list, merge_events = results

        # Cluster & rename objects to get globally unique event IDs
        split_merged_events_ds = self.cluster_rename_objects_and_props(
            object_id_field, object_props, overlap_objects_list, merge_events
        )
        
        # Rechunk final output
        chunk_dict = {
            self.timedim: self.timechunks,
            'ID': -1,
            'component': -1,
            'sibling_ID': -1,
            self.xdim: -1
        }
        if not self.unstructured_grid:
            chunk_dict[self.ydim] = -1
        
        split_merged_events_ds = split_merged_events_ds.chunk(chunk_dict)#.persist()
        if self.verbosity > 0:
            print('Finished clustering and renaming objects into coherent consistent events')
    
        # Count final number of events
        N_events = split_merged_events_ds.ID_field.max().compute().data
    
        return split_merged_events_ds, merge_events, N_events
    
    def cluster_rename_objects_and_props(self, object_id_field_unique, object_props, overlap_objects_list, merge_events):
        """
        Cluster the object pairs and relabel to determine final event IDs.
        
        Parameters
        ----------
        object_id_field_unique : xarray.DataArray
            Field of unique object IDs. IDs must not be repeated across time.
        object_props : xarray.Dataset
            Properties of each object that also need to be relabeled.
        overlap_objects_list : (N x 2) numpy.ndarray
            Array of object ID pairs that indicate which objects are in the same event. The object in the first column precedes the second column in time.
        merge_events : xarray.Dataset
            Information about merge events
            
        Returns
        -------
        split_merged_events_ds : xarray.Dataset
            Dataset with relabeled events and their properties. ID = 0 indicates no object.
        """
        ## Cluster the overlap_pairs into groups of IDs that are actually the same object
        # Get IDs from overlap pairs
        max_ID = object_id_field_unique.max().compute().values + 1
        IDs = np.arange(max_ID)
        
        # Convert overlap pairs to indices
        overlap_pairs_indices = np.array([(pair[0], pair[1]) for pair in overlap_objects_list])
        
        # Create a sparse matrix representation of the graph
        n = max_ID
        row_indices, col_indices = overlap_pairs_indices.T
        data = np.ones(len(overlap_pairs_indices), dtype=np.bool_)
        graph = csr_matrix((data, (row_indices, col_indices)), shape=(n, n), dtype=np.bool_)
        
        # Clean up temporary arrays
        del row_indices
        del col_indices
        del data
        
        # Solve the graph to determine connected components
        num_components, component_IDs = connected_components(csgraph=graph, directed=False, return_labels=True)
        
        del graph
        
        # Group IDs by their component
        ID_clusters = [[] for _ in range(num_components)]
        for ID, component_ID in zip(IDs, component_IDs):
            ID_clusters[component_ID].append(ID)

        ## ID_clusters now is a list of lists of equivalent object IDs that have been tracked across time
        #  We now need to replace all IDs in object_id_field_unique that match the equivalent_IDs with the list index:  This is the new/final ID field.
        
        # Create mapping from original IDs to cluster indices
        min_int32 = np.iinfo(np.int32).min
        max_old_ID = object_id_field_unique.max().compute().data
        ID_to_cluster_index_array = np.full(max_old_ID + 1, min_int32, dtype=np.int32)

        # Fill the lookup array
        for index, cluster in enumerate(ID_clusters):
            for ID in cluster:
                ID_to_cluster_index_array[ID] = np.int32(index)  # Because these are the connected IDs, there are many fewer!
                                                                 #  ID = 0 is still invalid/no object
        
        # Convert to DataArray for apply_ufunc
        #  N.B.: **Need to pass da into apply_ufunc, otherwise it doesn't manage the memory correctly with large shared-mem numpy arrays**
        ID_to_cluster_index_da = xr.DataArray(
            ID_to_cluster_index_array, 
            dims='ID', 
            coords={'ID': np.arange(max_old_ID + 1)}
        )
        
        def map_IDs_to_indices(block, ID_to_cluster_index_array):
            """Map original IDs to cluster indices."""
            mask = block > 0
            new_block = np.zeros_like(block, dtype=np.int32)
            new_block[mask] = ID_to_cluster_index_array[block[mask]]
            return new_block
        
        # Apply the mapping
        input_dims = [self.xdim] if self.unstructured_grid else [self.ydim, self.xdim]
        split_merged_relabeled_object_id_field = xr.apply_ufunc(
            map_IDs_to_indices,
            object_id_field_unique, 
            ID_to_cluster_index_da,
            input_core_dims=[input_dims, ['ID']],
            output_core_dims=[input_dims],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int32]
        ).persist()
        
        ## Relabel the object_props to match the new IDs (and add time dimension)
        
        max_new_ID = num_components + 1  # New IDs range from 0 to max_new_ID
        new_ids = np.arange(1, max_new_ID+1, dtype=np.int32)
        
        # Create new object_props dataset
        object_props_extended = xr.Dataset(coords={
            'ID': new_ids,
            self.timedim: object_id_field_unique[self.timedim]
        })
        
        # Create mapping from new IDs to the original IDs _at the corresponding time_
        valid_new_ids = (split_merged_relabeled_object_id_field > 0)      
        original_ids_field = object_id_field_unique.where(valid_new_ids)
        new_ids_field = split_merged_relabeled_object_id_field.where(valid_new_ids)
        
        if not self.unstructured_grid:
            original_ids_field = original_ids_field.stack(z=(self.ydim, self.xdim), create_index=False)
            new_ids_field = new_ids_field.stack(z=(self.ydim, self.xdim), create_index=False)
        
        new_id_to_idx = {id_val: idx for idx, id_val in enumerate(new_ids)}

        def process_timestep(orig_ids, new_ids_t):
            """Process a single timestep to create ID mapping."""
            result = np.zeros(len(new_id_to_idx), dtype=np.int32)
            
            valid_mask = new_ids_t > 0
            
            # Get valid points for this timestep
            if not valid_mask.any():
                return result
                
            orig_valid = orig_ids[valid_mask]
            new_valid = new_ids_t[valid_mask]
            
            if len(orig_valid) == 0:
                return result
                
            unique_pairs = np.unique(np.column_stack((orig_valid, new_valid)), axis=0)
            
            # Create mapping
            for orig_id, new_id in unique_pairs:
                if new_id in new_id_to_idx:
                    result[new_id_to_idx[new_id]] = orig_id
                    
            return result

        # Process in parallel
        input_dim = ['ncells'] if self.unstructured_grid else ['z']
        global_id_mapping = xr.apply_ufunc(
            process_timestep,
            original_ids_field,
            new_ids_field,
            input_core_dims=[input_dim, input_dim],
            output_core_dims=[['ID']],
            vectorize=True,
            dask='parallelized',
            output_dtypes=[np.int32],
            dask_gufunc_kwargs={'output_sizes': {'ID': len(new_ids)}}
        ).assign_coords(ID=new_ids).compute()
        
        # Store original ID mapping
        object_props_extended['global_ID'] = global_id_mapping
        # Post-condition: Now, e.g. global_id_mapping.sel(ID=10) --> Given the new ID (10), returns corresponding original_id at every time
        
        # Transfer all properties from original object_props
        dummy = object_props.isel(ID=0) * np.nan   # Add vale of ID = 0 to this coordinate ID
        object_props = xr.concat([dummy.assign_coords(ID=0), object_props], dim='ID')
        
        for var_name in object_props.data_vars:
            temp = (object_props[var_name]
                              .sel(ID=global_id_mapping.rename({'ID':'new_id'}))
                              .drop_vars('ID').rename({'new_id':'ID'}))
            
            if var_name == 'ID':
                temp = temp.astype(np.int32)
            else:
                temp = temp.astype(np.float32)
                
            object_props_extended[var_name] = temp
        
        ## Map the merge_events using the old IDs to be from dimensions (merge_ID, parent_idx) 
        #     --> new merge_ledger with dimensions (time, ID, sibling_ID)
        # i.e. for each merge_ID --> merge_parent_IDs   gives the old IDs  --> map to new ID using ID_to_cluster_index_da
        #                   --> merge_time
        
        old_parent_IDs = xr.where(merge_events.parent_IDs > 0, merge_events.parent_IDs, 0)
        new_IDs_parents = ID_to_cluster_index_da.sel(ID=old_parent_IDs)

        # Replace the coordinate merge_ID in new_IDs_parents with merge_time.  merge_events.merge_time gives merge_time for each merge_ID
        new_IDs_parents_t = (new_IDs_parents
                            .assign_coords({'merge_time': merge_events.merge_time})
                            .drop_vars('ID')
                            .swap_dims({'merge_ID': 'merge_time'})
                            .persist())
        
        # Map new_IDs_parents_t into a new data array with dimensions time, ID, and sibling_ID
        merge_ledger = (xr.full_like(global_id_mapping, fill_value=-1)
                       .chunk({self.timedim: self.timechunks})
                       .expand_dims({'sibling_ID': new_IDs_parents_t.parent_idx.shape[0]})
                       .copy())
        
        # Wrapper for processing/mapping mergers in parallel
        def process_time_group(time_block, IDs_data, IDs_coords):
            """Process all mergers for a single block of timesteps."""
            result = xr.full_like(time_block, -1)
            
            # Get unique times in this block
            unique_times = np.unique(time_block[self.timedim])
            
            for time_val in unique_times:
                # Get IDs for this time
                time_mask = IDs_coords['merge_time'] == time_val
                if not np.any(time_mask):
                    continue
                    
                IDs_at_time = IDs_data[time_mask]
                
                # Handle single merger case
                if IDs_at_time.ndim == 1:
                    valid_mask = IDs_at_time > 0
                    if np.any(valid_mask):
                        # Create expanded array for sibling_ID dimension
                        expanded_IDs = np.broadcast_to(
                            IDs_at_time, 
                            (len(time_block.sibling_ID), len(IDs_at_time))
                        )
                        result.loc[{self.timedim: time_val, 'ID': IDs_at_time[valid_mask]}] = expanded_IDs[:, valid_mask]
                
                # Handle multiple mergers case
                else:
                    for merger_IDs in IDs_at_time:
                        valid_mask = merger_IDs > 0
                        if np.any(valid_mask):
                            expanded_IDs = np.broadcast_to(
                                merger_IDs, 
                                (len(time_block.sibling_ID), len(merger_IDs))
                            )
                            result.loc[{self.timedim: time_val, 'ID': merger_IDs[valid_mask]}] = expanded_IDs[:, valid_mask]
                            
            return result
        
        # Map blocks in parallel
        merge_ledger = xr.map_blocks(
            process_time_group,
            merge_ledger,
            args=(new_IDs_parents_t.values, new_IDs_parents_t.coords),
            template=merge_ledger
        )

        # Format merge ledger
        merge_ledger = merge_ledger.rename('merge_ledger').transpose(self.timedim, 'ID', 'sibling_ID').persist()
        
        # For structured grid, convert centroid from pixel to lat/lon
        if not self.unstructured_grid:
            y_values = xr.DataArray(
                split_merged_relabeled_object_id_field[self.ydim].values, 
                coords=[np.arange(len(split_merged_relabeled_object_id_field[self.ydim]))], 
                dims=['pixels']
            )
            x_values = xr.DataArray(
                split_merged_relabeled_object_id_field[self.xdim].values, 
                coords=[np.arange(len(split_merged_relabeled_object_id_field[self.xdim]))], 
                dims=['pixels']
            )
            
            object_props_extended['centroid'] = xr.concat([
                y_values.interp(pixels=object_props_extended['centroid'].sel(component=0)),
                x_values.interp(pixels=object_props_extended['centroid'].sel(component=1))
            ], dim='component')
            
            object_props_extended = object_props_extended.drop_vars('pixels')
        
        # Add start and end time indices for each ID
        valid_presence = object_props_extended['global_ID'] > 0  # i.e. where there is valid data
        
        object_props_extended['presence'] = valid_presence
        object_props_extended['time_start'] = valid_presence.time[valid_presence.argmax(dim=self.timedim)]
        object_props_extended['time_end'] = valid_presence.time[
            (valid_presence.sizes[self.timedim] - 1) - (valid_presence[::-1]).argmax(dim=self.timedim)
        ]
                
        # Combine all components into final dataset
        split_merged_relabeled_events_ds = xr.merge([
            split_merged_relabeled_object_id_field.rename('ID_field'), 
            object_props_extended,
            merge_ledger
        ])
        
        # Remove the last ID -- it is all 0s
        return split_merged_relabeled_events_ds.isel(ID=slice(0, -1))
    
    # ============================
    # Splitting and Merging Methods
    # ============================
    
    def split_and_merge_objects(self, object_id_field_unique, object_props):
        """
        Implement object splitting and merging logic.
        
        This identifies and processes cases where objects split or merge over time,
        creating new object IDs as needed.
        
        Parameters
        ----------
        object_id_field_unique : xarray.DataArray
            Field of unique object IDs
        object_props : xarray.Dataset
            Properties of each object
            
        Returns
        -------
        tuple
            (object_id_field, object_props, overlap_objects_list, merge_events)
        """
        # Find overlapping objects
        overlap_objects_list = self.find_overlapping_objects(object_id_field_unique)  # List object pairs that overlap by at least overlap_threshold percent
        overlap_objects_list = self.enforce_overlap_threshold(overlap_objects_list, object_props)
        if self.verbosity > 0:
            print('Finished finding overlapping objects')
        
        # Initialise merge tracking lists
        merge_times = []      # When the merge occurred
        merge_child_ids = []  # Resulting child ID
        merge_parent_ids = [] # List of parent IDs that merged
        merge_areas = []      # Areas of overlap
        next_new_id = object_props.ID.max().item() + 1  # Start new IDs after highest existing ID
        
        # Find children (t+1 / RHS) that appear multiple times (merging objects) --> Indicates there are 2+ Parent Objects...
        unique_children, children_counts = np.unique(overlap_objects_list[:, 1], return_counts=True)
        merging_objects = unique_children[children_counts > 1]
        
        # Pre-compute time indices for each child object
        time_index_map = self.compute_id_time_dict(object_id_field_unique, merging_objects, next_new_id)
        Nx = object_id_field_unique[self.xdim].size
        
        # Group objects by time-chunk for efficient processing
        # Pre-condition: Object IDs should be monotonically increasing in time...
        chunk_boundaries = np.cumsum([0] + list(object_id_field_unique.chunks[0]))
        objects_by_chunk = {}
        # Ensure that objects_by_chunk has entry for every key
        for chunk_idx in range(len(object_id_field_unique.chunks[0])):
            objects_by_chunk.setdefault(chunk_idx, [])
        
        object_id_field_unique = object_id_field_unique.persist()
        
        # Assign objects to chunks
        for object_id in merging_objects:
            chunk_idx = np.searchsorted(chunk_boundaries, time_index_map[object_id], side='right') - 1
            objects_by_chunk.setdefault(chunk_idx, []).append(object_id)
        
        future_chunk_merges = []
        updated_chunks = []
        
        # Process each time chunk
        for chunk_idx, chunk_objects in objects_by_chunk.items():
            # We do this to avoid repetetively re-computing and injecting tiny changes into the full dask-backed DataArray object_id_field_unique
            
            # Extract and load an entire chunk into memory
            chunk_start = sum(object_id_field_unique.chunks[0][:chunk_idx])
            chunk_end = chunk_start + object_id_field_unique.chunks[0][chunk_idx] + 1 #  We also want access to the object_id_time_p1...  But need to remember to remove the last time later
            
            chunk_data = object_id_field_unique.isel({self.timedim: slice(chunk_start, chunk_end)}).compute()
            
            # Create a working queue of objects to process
            objects_to_process = chunk_objects.copy()
            # Combine only the future_chunk_merges that don't already appear in objects_to_process
            objects_to_process = objects_to_process + [
                object_id for object_id in future_chunk_merges 
                if object_id not in objects_to_process
            ]  # First, assess the new objects from the end of the previous chunk...
            future_chunk_merges = []
            
            # Process each object in this chunk
            while objects_to_process:  # Process until queue is empty
                child_id = objects_to_process.pop(0)
                
                # Get time index and data slices
                child_time_idx = time_index_map[child_id]
                relative_time_idx = child_time_idx - chunk_start
                
                object_id_time = chunk_data.isel({self.timedim: relative_time_idx})
                try:
                    object_id_time_p1 = chunk_data.isel({self.timedim: relative_time_idx+1})
                except:
                    # Last chunk
                    object_id_time_p1 = xr.full_like(object_id_time, 0)
                
                # Get previous timestep
                if relative_time_idx-1 >= 0:
                    object_id_time_m1 = chunk_data.isel({self.timedim: relative_time_idx-1})
                elif updated_chunks:
                    # Get the last time slice from the previous chunk (stored in updated_chunks)
                    _, _, last_chunk_data = updated_chunks[-1]
                    object_id_time_m1 = last_chunk_data[-1]
                else:
                    object_id_time_m1 = xr.full_like(object_id_time, 0)
                
                # Get mask of child object
                child_mask_2d = (object_id_time == child_id).values
                
                # Find all pairs involving this child
                child_mask = overlap_objects_list[:, 1] == child_id
                child_where = np.where(overlap_objects_list[:, 1] == child_id)[0]
                merge_group = overlap_objects_list[child_mask]
                
                # Get parent objects (LHS) that overlap with this child object
                parent_ids = merge_group[:, 0]
                num_parents = len(parent_ids)
                
                # Create new IDs for the other half of the child object & record in the merge ledger
                new_object_id = np.arange(next_new_id, next_new_id + (num_parents - 1), dtype=np.int32)
                next_new_id += num_parents - 1
                
                # Replace the 2nd+ child in the overlap objects list with the new child ID
                overlap_objects_list[child_where[1:], 1] = new_object_id
                child_ids = np.concatenate((np.array([child_id]), new_object_id))
                
                # Record merge event
                merge_times.append(chunk_data.isel({self.timedim: relative_time_idx}).time.values)
                merge_child_ids.append(child_ids)
                merge_parent_ids.append(parent_ids)
                merge_areas.append(overlap_objects_list[child_mask, 2])
                
                ## Relabel the Original Child Object ID Field to account for the New ID:
                # Get parent centroids for partitioning
                parent_centroids = object_props.sel(ID=parent_ids).centroid.values.T
                
                # Partition the child object based on parent associations
                if self.nn_partitioning: 
                    # Nearest-neighbor partitioning
                    # --> For every (Original) Child Cell in the ID Field, Find the closest (t-1) Parent _Cell_
                    if self.unstructured_grid:
                        # Prepare parent masks
                        parent_masks = np.zeros((len(parent_ids), object_id_time.shape[0]), dtype=bool)
                        for idx, parent_id in enumerate(parent_ids):
                            parent_masks[idx] = (object_id_time_m1 == parent_id).values
                        
                        # Calculate maximum search distance
                        max_area = np.max(object_props.sel(ID=parent_ids).area.values) / self.mean_cell_area
                        max_distance = int(np.sqrt(max_area) * 2.0)
                        
                        # Use optimised unstructured partitioning
                        new_labels = partition_nn_unstructured(
                            child_mask_2d,
                            parent_masks,
                            child_ids,
                            parent_centroids,
                            self.neighbours_int.values,
                            self.lat.values,  # Need to pass these as NumPy arrays for JIT compatibility
                            self.lon.values,
                            max_distance=max(max_distance, 20)*2  # Set minimum threshold, in cells
                        )
                    else:
                        # Prepare parent masks for structured grid
                        parent_masks = np.zeros((len(parent_ids), object_id_time.shape[0], object_id_time.shape[1]), dtype=bool)
                        for idx, parent_id in enumerate(parent_ids):
                            parent_masks[idx] = (object_id_time_m1 == parent_id).values
                        
                        # Calculate maximum search distance
                        max_area = np.max(object_props.sel(ID=parent_ids).area.values)
                        max_distance = int(np.sqrt(max_area) * 3.0)  # Use 3x the max blob radius
                        
                        # Use optimised structured grid partitioning
                        new_labels = partition_nn_grid(
                            child_mask_2d,
                            parent_masks, 
                            child_ids,
                            parent_centroids,
                            Nx,
                            max_distance=max(max_distance, 40)  # Set minimum threshold, in cells
                        )
                        
                else:
                    # Centroid-based partitioning
                    # --> For every (Original) Child Cell in the ID Field, Find the closest (t-1) Parent _Centroid_
                    if self.unstructured_grid:
                        new_labels = partition_centroid_unstructured(
                            child_mask_2d,
                            parent_centroids,
                            child_ids,
                            self.lat.values,
                            self.lon.values
                        )                      
                    else:
                        # Calculate distances to each parent centroid
                        distances = wrapped_euclidian_parallel(child_mask_2d, parent_centroids, Nx)
                        
                        # Assign based on closest parent
                        new_labels = child_ids[np.argmin(distances, axis=1)]
                
                ## Update values in child_time_idx and assign the updated slice back to the original DataArray
                temp = np.zeros_like(object_id_time)
                temp[child_mask_2d] = new_labels
                object_id_time = object_id_time.where(~child_mask_2d, temp)
                chunk_data[{self.timedim: relative_time_idx}] = object_id_time
                
                # Add new entries to time_index_map for each of new_object_id corresponding to the current time index
                time_index_map.update({new_id: child_time_idx for new_id in new_object_id})
                
                # Update the Properties of the N Children Objects
                new_child_props = self.calculate_object_properties(object_id_time, properties=['area', 'centroid'])
                
                # Update the object_props DataArray:  (but first, check if the original children still exists)
                if child_id in new_child_props.ID:
                    # Update existing entry
                    object_props.loc[dict(ID=child_id)] = new_child_props.sel(ID=child_id)
                else:  # Delete child_id:  The object has split/morphed such that it doesn't get a partition of this child...
                    object_props = object_props.drop_sel(ID=child_id)  # N.B.: This means that the IDs are no longer continuous...
                    if self.verbosity > 0:
                        print(f"Deleted child_id {child_id} because parents have split/morphed")
                
                # Add the properties for the N-1 other new child ID
                new_object_ids_still = new_child_props.ID.where(
                    new_child_props.ID.isin(new_object_id), drop=True
                ).ID
                object_props = xr.concat([object_props, new_child_props.sel(ID=new_object_ids_still)], dim='ID')
                
                missing_ids = set(new_object_id) - set(new_object_ids_still.values)
                if len(missing_ids) > 0 and self.verbosity > 0:
                    print(f"Missing newly created child_ids {missing_ids} because parents have split/morphed in the meantime...")

                ## Finally, Re-assess all of the Parent IDs (LHS) equal to the (original) child_id
                
                # Look at the overlap IDs between the original child_id and the next time-step, and also the new_object_id and the next time-step
                new_overlaps = self.check_overlap_slice(object_id_time.values, object_id_time_p1.values)
                new_child_overlaps_list = new_overlaps[(new_overlaps[:, 0] == child_id) | np.isin(new_overlaps[:, 0], new_object_id)]
                new_child_overlaps_list = self.enforce_overlap_threshold(new_child_overlaps_list, object_props)
                
                # Replace the lines in the overlap_objects_list where (original) child_id is on the LHS, with these new pairs in new_child_overlaps_list
                child_mask_LHS = overlap_objects_list[:, 0] == child_id
                overlap_objects_list = np.concatenate([
                    overlap_objects_list[~child_mask_LHS], 
                    new_child_overlaps_list
                ])
                
                ## Finally, _FINALLY_, we need to ensure that of the new children objects we made, they only overlap with their respective parent...
                new_unique_children, new_children_counts = np.unique(
                    new_child_overlaps_list[:, 1], 
                    return_counts=True
                )
                new_merging_objects = new_unique_children[new_children_counts > 1]
                
                if new_merging_objects.size > 0:
                    if relative_time_idx + 1 < chunk_data.sizes[self.timedim] - 1:  # If there is a next time-step in this chunk
                        # Add to current queue if in this chunk
                        for new_child_id in new_merging_objects:
                            if new_child_id not in objects_to_process: # We aren't already going to assess this object
                                objects_to_process.insert(0, new_child_id)
                    else:  # This is out of our current jurisdiction: Defer this reassessment to the beginning of the next chunk
                        # Add to next chunk's queue
                        future_chunk_merges.extend(new_merging_objects)
            
            # Store the processed chunk
            updated_chunks.append((chunk_start, chunk_end-1, chunk_data[:(chunk_end-1-chunk_start)]))
            
            if chunk_idx % 10 == 0 and self.verbosity > 0:
                print(f"Processing splitting and merging in chunk {chunk_idx} of {len(objects_by_chunk)}")
                
                # Periodically update main array to manage memory
                if len(updated_chunks) > 1:  # Keep the last chunk for potential object_id_time_m1 reference
                    for start, end, chunk_data in updated_chunks[:-1]:
                        object_id_field_unique[{self.timedim: slice(start, end)}] = chunk_data
                    updated_chunks = updated_chunks[-1:]  # Keep only the last chunk
                    object_id_field_unique = object_id_field_unique.persist()
        
        # Apply final chunk updates
        for start, end, chunk_data in updated_chunks:
            object_id_field_unique[{self.timedim: slice(start, end)}] = chunk_data
        object_id_field_unique = object_id_field_unique.persist()
        
        # Process merge events into a dataset
        max_parents = max(len(ids) for ids in merge_parent_ids)
        max_children = max(len(ids) for ids in merge_child_ids)
        
        # Convert lists to padded numpy arrays
        parent_ids_array = np.full((len(merge_parent_ids), max_parents), -1, dtype=np.int32)
        child_ids_array = np.full((len(merge_child_ids), max_children), -1, dtype=np.int32)
        overlap_areas_array = np.full((len(merge_areas), max_parents), -1, dtype=np.int32)
        
        for i, parents in enumerate(merge_parent_ids):
            parent_ids_array[i, :len(parents)] = parents
        
        for i, children in enumerate(merge_child_ids):
            child_ids_array[i, :len(children)] = children
        
        for i, areas in enumerate(merge_areas):
            overlap_areas_array[i, :len(areas)] = areas
        
        # Create merge events dataset
        merge_events = xr.Dataset(
            {
                'parent_IDs': (('merge_ID', 'parent_idx'), parent_ids_array),
                'child_IDs': (('merge_ID', 'child_idx'), child_ids_array),
                'overlap_areas': (('merge_ID', 'parent_idx'), overlap_areas_array),
                'merge_time': ('merge_ID', merge_times),
                'n_parents': ('merge_ID', np.array([len(p) for p in merge_parent_ids], dtype=np.int8)),
                'n_children': ('merge_ID', np.array([len(c) for c in merge_child_ids], dtype=np.int8))
            },
            attrs={
                'fill_value': -1
            }
        )
        
        object_props = object_props.persist()
        
        return (object_id_field_unique, 
                object_props, 
                overlap_objects_list[:, :2],
                merge_events)

    
    def split_and_merge_objects_parallel(self, object_id_field_unique, object_props):
        """
        Optimised parallel implementation of object splitting and merging.
        
        This version is specifically designed for unstructured grids with more efficient 
        memory handling and better parallelism than the standard split_and_merge_objects 
        method. It processes data in chunks, handles merging events, and efficiently 
        updates object IDs.
        
        Parameters
        ----------
        object_id_field_unique : xarray.DataArray
            Field of unique object IDs
        object_props : xarray.Dataset
            Properties of each object
            
        Returns
        -------
        tuple
            (object_id_field, object_props, overlap_objects_list, merge_events)
        """
        
        # Constants for memory allocation
        MAX_MERGES = 20   # Maximum number of merges per timestep
        MAX_PARENTS = 10  # Maximum number of parents per merge
        MAX_CHILDREN = MAX_PARENTS
                
        def process_chunk(chunk_data_m1_full, chunk_data_p1_full, merging_objects, next_id_start, lat, lon, area, neighbours_int):
            """
            Process a single chunk of merging objects.
            
            This function handles the complex batch processing of splitting and merging objects 
            across timesteps within a single chunk. It finds overlapping objects, determines
            parent-child relationships, and creates new IDs as needed.
            
            Parameters
            ----------
            chunk_data_m1_full : numpy.ndarray
                Data from previous timestep (t-1) and current timestep (t)
            chunk_data_p1_full : numpy.ndarray
                Data from next timestep (t+1)
            merging_objects : (n_time, max_merges) numpy.ndarray
                IDs of objects to process
            next_id_start : (n_time, max_merges) numpy.ndarray
                Starting ID values for new objects
            lat, lon : numpy.ndarray
                Latitude/longitude arrays
            area : numpy.ndarray
                Cell area array
            neighbours_int : numpy.ndarray
                Neighbor connectivity array
            
            Returns
            -------
            tuple
                Contains merge events, object updates, and newly created objects
            """
            
            ## Fix Broadcasted dimensions of inputs: 
            #    Remove extra dimension if present while preserving time chunks
            #    N.B.: This is a weird artefact/choice of xarray apply_ufunc broadcasting... (i.e. 'nv' dimension gets injected into all the other arrays!)
            
            chunk_data_m1 = chunk_data_m1_full.squeeze()[0].astype(np.int32).copy()
            chunk_data = chunk_data_m1_full.squeeze()[1].astype(np.int32).copy()
            del chunk_data_m1_full  # Free memory immediately
            chunk_data_p1 = chunk_data_p1_full.squeeze().astype(np.int32).copy()
            del chunk_data_p1_full
            
            # Extract and prepare input arrays
            lat = lat.squeeze().astype(np.float32)
            lon = lon.squeeze().astype(np.float32)
            area = area.squeeze().astype(np.float32)
            next_id_start = next_id_start.squeeze()
            
            # Handle neighbours_int with correct dimensions (nv, ncells)
            neighbours_int = neighbours_int.squeeze()
            if neighbours_int.shape[1] != lat.shape[0]:
                neighbours_int = neighbours_int.T
            
            # Handle multiple merging objects - ensure proper dimensionality
            merging_objects = merging_objects.squeeze()
            if merging_objects.ndim == 1:
                merging_objects = merging_objects[:, None]  # Add dimension for max_merges
            
            # Pre-convert lat/lon to Cartesian coordinates for efficiency
            x = (np.cos(np.radians(lat)) * np.cos(np.radians(lon))).astype(np.float32)
            y = (np.cos(np.radians(lat)) * np.sin(np.radians(lon))).astype(np.float32)
            z = np.sin(np.radians(lat)).astype(np.float32)
            
            # Pre-allocate output arrays
            n_time = chunk_data_p1.shape[0]
            n_points = chunk_data_p1.shape[1]
            
            merge_child_ids = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.int32)
            merge_parent_ids = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.int32)
            merge_areas = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.float32)
            merge_counts = np.zeros(n_time, dtype=np.int16)  # Number of merges per timestep

            updates_array = np.full((n_time, n_points), 255, dtype=np.uint8)
            updates_ids = np.full((n_time, 255), -1, dtype=np.int32)
            has_merge = np.zeros(n_time, dtype=np.bool_)
            
            # Prepare merging objects list for each timestep
            merging_objects_list = [list(merging_objects[i][merging_objects[i] > 0]) 
                                for i in range(merging_objects.shape[0])]
            final_merging_objects = np.full((n_time, MAX_MERGES), -1, dtype=np.int32)
            final_merge_count = 0
            
            # Process each timestep
            for t in range(n_time):
                next_new_id = next_id_start[t]  # Use the offset for this timestep
                
                # Get current time slice data
                if t == 0:
                    data_m1 = chunk_data_m1
                    data_t = chunk_data
                    del chunk_data_m1, chunk_data  # Free memory
                else:
                    data_m1 = data_t
                    data_t = data_p1
                data_p1 = chunk_data_p1[t]
                
                # Process each merging object at this timestep
                while merging_objects_list[t]:
                    child_id = merging_objects_list[t].pop(0)
                    
                    # Get child mask and identify overlapping parents
                    child_mask = (data_t == child_id)
                    
                    # Find parent objects that overlap with this child
                    potential_parents = np.unique(data_m1[child_mask])
                    parent_iterator = 0
                    parent_masks_uint = np.full(n_points, 255, dtype=np.uint8)
                    parent_centroids = np.full((MAX_PARENTS, 2), -1.e10, dtype=np.float32)
                    parent_ids = np.full(MAX_PARENTS, -1, dtype=np.int32)
                    parent_areas = np.zeros(MAX_PARENTS, dtype=np.float32)
                    overlap_areas = np.zeros(MAX_PARENTS, dtype=np.float32)
                    n_parents = 0
                    
                    # Find all unique parent IDs with significant overlap
                    for parent_id in potential_parents[potential_parents > 0]:
                        if n_parents >= MAX_PARENTS:
                            raise RuntimeError(f"Reached maximum number of parents ({MAX_PARENTS}) for child {child_id} at timestep {t}")
                            
                        parent_mask = (data_m1 == parent_id)
                        if np.any(parent_mask & child_mask):
                            # Calculate overlap area and check if it's large enough
                            area_0 = area[parent_mask].sum()  # Parent area
                            area_1 = area[child_mask].sum()   # Child area
                            min_area = np.minimum(area_0, area_1)
                            overlap_area = area[parent_mask & child_mask].sum()
                            
                            # Skip if overlap is below threshold
                            if overlap_area / min_area < self.overlap_threshold:
                                continue
                            
                            # Record parent information
                            parent_masks_uint[parent_mask] = parent_iterator
                            parent_ids[n_parents] = parent_id
                            overlap_areas[n_parents] = overlap_area
                            
                            # Calculate area-weighted centroid for this parent
                            mask_area = area[parent_mask]
                            weighted_coords = np.array([
                                np.sum(mask_area * x[parent_mask]),
                                np.sum(mask_area * y[parent_mask]),
                                np.sum(mask_area * z[parent_mask])
                            ], dtype=np.float32)
                            
                            norm = np.sqrt(np.sum(weighted_coords * weighted_coords))
                                        
                            # Convert back to lat/lon
                            parent_centroids[n_parents, 0] = np.degrees(np.arcsin(weighted_coords[2]/norm))
                            parent_centroids[n_parents, 1] = np.degrees(np.arctan2(weighted_coords[1], weighted_coords[0]))
                            
                            # Fix longitude range to [-180, 180]
                            if parent_centroids[n_parents, 1] > 180:
                                parent_centroids[n_parents, 1] -= 360
                            elif parent_centroids[n_parents, 1] < -180:
                                parent_centroids[n_parents, 1] += 360
                            
                            parent_areas[n_parents] = area_0
                            parent_iterator += 1
                            n_parents += 1
                    
                    # Need at least 2 parents for merging
                    if n_parents < 2:
                        continue
                    
                    # Create new IDs for each partition
                    new_child_ids = np.arange(next_new_id, next_new_id + (n_parents - 1), dtype=np.int32)
                    child_ids = np.concatenate((np.array([child_id]), new_child_ids))
                    
                    # Record merge event
                    curr_merge_idx = merge_counts[t]
                    if curr_merge_idx > MAX_MERGES:
                        raise RuntimeError(f"Reached maximum number of merges ({MAX_MERGES}) at timestep {t}")
                    
                    merge_child_ids[t, curr_merge_idx, :n_parents] = child_ids[:n_parents]
                    merge_parent_ids[t, curr_merge_idx, :n_parents] = parent_ids[:n_parents]
                    merge_areas[t, curr_merge_idx, :n_parents] = overlap_areas[:n_parents]
                    merge_counts[t] += 1
                    has_merge[t] = True
                    
                    # Partition the child object based on parent associations
                    if self.nn_partitioning:
                        # Estimate maximum search distance based on object size
                        max_area = parent_areas.max() / self.mean_cell_area
                        max_distance = int(np.sqrt(max_area) * 2.0)
                        
                        # Use optimised nearest-neighbor partitioning
                        new_labels_uint = partition_nn_unstructured_optimised(
                            child_mask.copy(),
                            parent_masks_uint.copy(),
                            parent_centroids,
                            neighbours_int.copy(),
                            lat,
                            lon,
                            max_distance=max(max_distance, 20)*2
                        )
                        # Returned 'new_labels_uint' is just the index of the child_ids
                        new_labels = child_ids[new_labels_uint]
                        
                        # Help garbage collection
                        new_labels_uint = None
                        
                    else:
                        # Use centroid-based partitioning
                        new_labels = partition_centroid_unstructured(
                            child_mask,
                            parent_centroids,
                            child_ids,
                            lat,
                            lon
                        )
                    
                    # Update slice data for subsequent merging in process_chunk
                    data_t[child_mask] = new_labels
                    
                    # Record which cells get which new IDs for later updates
                    spatial_indices_all = np.where(child_mask)[0]
                    child_mask = None  # Free memory
                    gc.collect()
                    
                    # Record update information for each new ID
                    for new_id in child_ids[1:]:
                        update_idx = np.where(updates_ids[t] == -1)[0][0]  # Find next non-negative index in updates_ids
                        updates_ids[t, update_idx] = new_id
                        updates_array[t, spatial_indices_all[new_labels == new_id]] = update_idx
                    
                    next_new_id += n_parents - 1
                    
                    # Find all child objects in the next timestep that overlap with our newly labeled regions
                    new_merging_list = []
                    for new_id in child_ids:
                        parent_mask = (data_t == new_id)
                        if np.any(parent_mask):
                            area_0 = area[parent_mask].sum()
                            potential_children = np.unique(data_p1[parent_mask])
                            
                            for potential_child in potential_children[potential_children > 0]:
                                potential_child_mask = (data_p1 == potential_child)
                                area_1 = area[potential_child_mask].sum()
                                min_area = min(area_0, area_1)
                                overlap_area = area[parent_mask & potential_child_mask].sum()
                                
                                if overlap_area / min_area > self.overlap_threshold:
                                    new_merging_list.append(potential_child)
                    
                    # Add newly found merging objects to processing queue
                    if t < n_time - 1:
                        # Add to next timestep in this chunk
                        for new_object_id in new_merging_list:
                            if new_object_id not in merging_objects_list[t+1]:
                                merging_objects_list[t+1].append(new_object_id)
                    else:
                        # Record for next chunk
                        for new_object_id in new_merging_list:
                            if final_merge_count > MAX_MERGES:
                                raise RuntimeError(f"Reached maximum number of merges ({MAX_MERGES}) at timestep {t}")
                            
                            if not np.any(final_merging_objects[t][:final_merge_count] == new_object_id):
                                final_merging_objects[t][final_merge_count] = new_object_id
                                final_merge_count += 1
                    
                            
            return (merge_child_ids, merge_parent_ids, merge_areas, merge_counts, 
                    has_merge, updates_array, updates_ids, final_merging_objects)
        

        def update_object_id_field_inplace(object_id_field, id_lookup, updates_array, updates_ids, has_merge):
            """
            Update the object field with chunk results using xarray operations.
            
            This is memory efficient as it avoids creating full copies of the object_id_field.
            
            Parameters
            ----------
            object_id_field : xarray.DataArray
                The full object field to update
            id_lookup : dict
                Dictionary mapping temporary IDs to new IDs
            updates_array : xarray.DataArray
                Array indicating which spatial indices to update
            updates_ids : xarray.DataArray
                The new IDs to assign to updated indices
            has_merge : xarray.DataArray
                Boolean indicating whether each timestep has merges
            
            Returns
            -------
            xarray.DataArray
                Updated object field
            """
            
            # Quick return if no merges to update
            if not has_merge.any():
                return object_id_field
            
            def update_timeslice(data, updates, update_ids, lookup_values):
                """Process a single timeslice."""
                # Extract valid update IDs
                valid_ids = update_ids[update_ids > -1]
                if len(valid_ids) == 0:
                    return data
                    
                # Create result array starting with original values
                result = data.copy()
                
                # Apply each update
                for idx, update_id in enumerate(valid_ids):
                    mask = updates == idx
                    if mask.any():
                        result = np.where(mask, lookup_values[update_id], result)
                        
                return result
            
            # Convert lookup dict to array for vectorized access
            max_id = max(id_lookup.keys()) + 1
            lookup_array = np.full(max_id, -1, dtype=np.int32)
            for temp_id, new_id in id_lookup.items():
                lookup_array[temp_id] = new_id
            
            # Apply updates in parallel
            result = xr.apply_ufunc(
                update_timeslice,
                object_id_field,
                updates_array,
                updates_ids,
                kwargs={'lookup_values': lookup_array},
                input_core_dims=[[self.xdim],
                                [self.xdim],
                                ['update_idx']],
                output_core_dims=[[self.xdim]],
                vectorize=True, 
                dask='parallelized',
                output_dtypes=[np.int32]
            )
            
            return result
        
        def update_object_id_field_zarr(object_id_field, id_lookup, updates_array, updates_ids, has_merge):
            """
            Update object field using a temporary zarr store for better memory efficiency.
            
            This approach minimises memory usage by writing changes directly to disk,
            allowing for more efficient parallel processing of large datasets.
            
            Parameters
            ----------
            object_id_field : xarray.DataArray
                The object field to update
            id_lookup : dict
                Dictionary mapping temporary IDs to new IDs
            updates_array : xarray.DataArray
                Array indicating which spatial indices to update
            updates_ids : xarray.DataArray
                The new IDs to assign to updated indices
            has_merge : xarray.DataArray
                Boolean indicating whether each timestep has merges
                
            Returns
            -------
            xarray.DataArray
                Updated object field from zarr store
            """
            
            # Early return if no merges to save memory
            if not has_merge.any().compute().item():
                return object_id_field
            
            zarr_path = f'{self.scratch_dir}/marEx_temp_field.zarr/'
            
            # Initialise zarr store if needed
            if not os.path.exists(zarr_path):
                object_id_field.name = 'temp'
                object_id_field.to_zarr(zarr_path, mode='w')
            
            def update_time_chunk(ds_chunk, lookup_dict):
                """Process a single chunk with optimised memory usage."""
                
                # Skip processing if no merges in this chunk
                needs_update = ds_chunk['has_merge'].any().compute().item()
                if not needs_update:
                    return ds_chunk['object_field']
                
                # Extract data from the chunk
                chunk_data = ds_chunk['object_field']
                chunk_updates = ds_chunk['updates']
                chunk_update_ids = ds_chunk['update_ids']
                
                # Get zarr region indices
                time_idx_start = int(ds_chunk['time_indices'].values[0])
                time_idx_end = int(ds_chunk['time_indices'].values[-1]) + 1
                
                updated_chunk = chunk_data.copy()
                
                # Process each time slice in the chunk
                for t in range(chunk_data.sizes[self.timedim]):
                    # Get update information for this time
                    updates_slice = chunk_updates.isel({self.timedim: t}).values
                    update_ids_slice = chunk_update_ids.isel({self.timedim: t}).values
                    
                    # Get valid update IDs
                    valid_mask = update_ids_slice > -1
                    if not np.any(valid_mask):
                        continue
                        
                    valid_ids = update_ids_slice[valid_mask]
                    
                    # Get the time slice data and apply updates
                    result_slice = updated_chunk.isel({self.timedim: t})
                    
                    for idx, update_id in enumerate(valid_ids):
                        mask = updates_slice == idx
                        if np.any(mask):
                            new_id = lookup_dict.get(int(update_id), update_id)
                            result_slice = xr.where(mask, new_id, result_slice)
                    
                    # Store updated slice
                    updated_chunk[t] = result_slice
                
                # Write the updated chunk directly to zarr
                updated_chunk.name = 'temp'
                updated_chunk.to_zarr(zarr_path, 
                                    region={self.timedim: slice(time_idx_start, time_idx_end)})
                
                return chunk_data  # Return original data for dask graph consistency
            
            # Create time indices for slicing
            time_coords = object_id_field[self.timedim].values
            time_indices = np.arange(len(time_coords))
            time_index_da = xr.DataArray(time_indices, dims=[self.timedim], coords={self.timedim: time_coords})
            
            # Create dataset with all necessary components
            ds = xr.Dataset({
                'object_field': object_id_field,
                'updates': updates_array,
                'update_ids': updates_ids,
                'time_indices': time_index_da,
                'has_merge': has_merge
            }).chunk({self.timedim: self.timechunks})
            
            # Process chunks in parallel
            result = xr.map_blocks(
                update_time_chunk,
                ds,
                kwargs={"lookup_dict": id_lookup},
                template=object_id_field
            )
            
            # Force computation to ensure all writes complete
            result = result.persist()
            wait(result)
            
            # Release resources
            del result, ds, object_id_field
            gc.collect()
            
            # Load the updated data from zarr store
            object_id_field_new = xr.open_zarr(zarr_path, chunks={self.timedim: self.timechunks}).temp
            
            return object_id_field_new
        
        
        def merge_objects_parallel_iteration(object_id_field_unique, merging_objects, global_id_counter):
            """
            Perform a single iteration of the parallel merging process.
            
            This function handles one complete batch of merging objects across all 
            timesteps, updating object IDs and tracking merge events.
            
            Parameters
            ----------
            object_id_field_unique : xarray.DataArray
                Field of unique object IDs
            merging_objects : set
                Set of object IDs to process in this iteration
            global_id_counter : int
                Current counter for assigning new global IDs
                
            Returns
            -------
            tuple
                (updated_field, merge_data, new_merging_objects, updated_counter)
            """
            
            n_time = len(object_id_field_unique[self.timedim])
            
            # Pre-allocate arrays for this iteration
            child_ids_iter = np.full((n_time, MAX_MERGES, MAX_CHILDREN), -1, dtype=np.int32)     # List of child ID arrays for this time
            parent_ids_iter = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.int32)     # List of parent ID arrays for this time
            merge_areas_iter = np.full((n_time, MAX_MERGES, MAX_PARENTS), -1, dtype=np.float32)  # List of areas for this time
            merge_counts_iter = np.zeros(n_time, dtype=np.int32)
            
            # Prepare neighbour information
            neighbours_int = self.neighbours_int.chunk({self.xdim: -1, 'nv': -1})
            
            if self.verbosity > 0:
                print(f"Processing Parallel Iteration {iteration + 1} with {len(merging_objects)} Merging Objects...")
            
            # Pre-compute the child_time_idx for merging_objects
            time_index_map = self.compute_id_time_dict(object_id_field_unique, list(merging_objects), global_id_counter)
            if self.verbosity > 1:
                print('  Finished Mapping Children to Time Indices')
            
            # Create uniform array of merging objects for each timestep
            max_merges = max(len([b for b in merging_objects if time_index_map.get(b, -1) == t]) for t in range(n_time))
            uniform_merging_objects_array = np.zeros((n_time, max_merges), dtype=np.int64)
            for t in range(n_time):
                objects_at_t = [b for b in merging_objects if time_index_map.get(b, -1) == t]
                if objects_at_t:  # Only fill if there are objects at this time
                    uniform_merging_objects_array[t, :len(objects_at_t)] = np.array(objects_at_t, dtype=np.int64)

            # Create DataArrays for parallel processing
            merging_objects_da = xr.DataArray(
                uniform_merging_objects_array,
                dims=[self.timedim, 'merges'],
                coords={self.timedim: object_id_field_unique[self.timedim]}
            )
            
            # Calculate ID offsets for each timestep to ensure unique IDs
            next_id_offsets = np.arange(n_time) * max_merges * self.timechunks + global_id_counter
            # N.B.: We also need to account for possibility of newly-split objects then creating more than max_merges by the end of the iteration through the chunk
            #         !!! This is likely the root cause of any errors such as "ID needs to be contiguous/continuous/full/unrepeated"
            next_id_offsets_da = xr.DataArray(
                next_id_offsets,
                dims=[self.timedim],
                coords={self.timedim: object_id_field_unique[self.timedim]}
            )
            
            # Create shifted arrays for time connectivity
            object_id_field_unique_p1 = object_id_field_unique.shift({self.timedim: -1}, fill_value=0)
            object_id_field_unique_m1 = object_id_field_unique.shift({self.timedim: 1}, fill_value=0)
            
            # Align chunks for better parallel processing
            object_id_field_unique_m1 = object_id_field_unique_m1.chunk({self.timedim: self.timechunks})
            object_id_field_unique_p1 = object_id_field_unique_p1.chunk({self.timedim: self.timechunks})
            merging_objects_da = merging_objects_da.chunk({self.timedim: self.timechunks})
            next_id_offsets_da = next_id_offsets_da.chunk({self.timedim: self.timechunks})
            
            # Process chunks in parallel
            results = xr.apply_ufunc(
                process_chunk,
                object_id_field_unique_m1,
                object_id_field_unique_p1,
                merging_objects_da,
                next_id_offsets_da,
                self.lat,
                self.lon,
                self.cell_area,
                neighbours_int,
                input_core_dims=[
                    [self.xdim], [self.xdim], ['merges'], [], 
                    [self.xdim], [self.xdim], [self.xdim], ['nv', self.xdim]
                ],
                output_core_dims=[
                    ['merge', 'parent'], ['merge', 'parent'], ['merge', 'parent'], [],
                    [], [self.xdim], ['update_idx'], ['merge']
                ],
                output_dtypes=[
                    np.int32, np.int32, np.float32, np.int16, 
                    np.bool_, np.uint8, np.int32, np.int32
                ],
                dask_gufunc_kwargs={
                    'output_sizes': {
                        'merge': MAX_MERGES,
                        'parent': MAX_PARENTS,
                        'update_idx': 255
                    }
                },
                vectorize=False,
                dask='parallelized'
            )

            # Unpack and persist results
            (merge_child_ids, merge_parent_ids, merge_areas, merge_counts,
                has_merge, updates_array, updates_ids, final_merging_objects) = results
            
            results = persist(
                merge_child_ids, merge_parent_ids, merge_areas, merge_counts,
                has_merge, updates_array, updates_ids, final_merging_objects
            )
            (merge_child_ids, merge_parent_ids, merge_areas, merge_counts, 
                has_merge, updates_array, updates_ids, final_merging_objects) = results
            
            # Get time indices where merges occurred
            has_merge = has_merge.compute()
            time_indices = np.where(has_merge)[0]
            
            # Clean up temporary arrays to save memory
            del object_id_field_unique_p1, object_id_field_unique_m1, merging_objects_da, next_id_offsets_da
            gc.collect()
            
            if self.verbosity > 1:
                print('  Finished Batch Processing Step')
            
            
            # ====== Global Consolidation of Data ======
            
            # 1. Collect all temporary IDs and create global mapping
            all_temp_ids = np.unique(
                merge_child_ids.where(merge_child_ids >= global_id_counter, other=0).compute().values
            )
            all_temp_ids = all_temp_ids[all_temp_ids > 0]  # Remove the 0
            
            if not len(all_temp_ids):  # If no temporary IDs exist
                id_lookup = {}
            else:            
                # Create mapping from temporary to permanent IDs
                id_lookup = {
                    temp_id: np.int32(new_id) for temp_id, new_id in zip(
                        all_temp_ids,
                        range(global_id_counter, global_id_counter + len(all_temp_ids))
                    )
                }
                global_id_counter += len(all_temp_ids)
            
            if self.verbosity > 1:
                print('  Finished Consolidation Step 1: Temporary ID Mapping')
            
            # 2. Update object ID field with new IDs
            update_on_disk = True  # This is more memory efficient because it refreshes the dask graph every iteration
            
            if update_on_disk:
                object_id_field_unique = update_object_id_field_zarr(
                    object_id_field_unique, id_lookup, updates_array, updates_ids, has_merge
                )
            else:
                object_id_field_unique = update_object_id_field_inplace(
                    object_id_field_unique, id_lookup, updates_array, updates_ids, has_merge
                )
                object_id_field_unique = object_id_field_unique.chunk({self.timedim: self.timechunks}) # Rechunk to avoid accumulating chunks...
            
            # Clean up arrays no longer needed
            del updates_array, updates_ids
            gc.collect()
            
            if self.verbosity > 1:
                print('  Finished Consolidation Step 2: Data Field Update')
            
            # 3. Update merge events
            new_merging_objects = set()
            merge_counts = merge_counts.compute()
            
            for t in time_indices:
                count = merge_counts.isel({self.timedim: t}).item()
                if count > 0:
                    merge_counts_iter[t] = count
                    
                    # Extract valid IDs and areas for each merge event
                    for merge_idx in range(count):
                        # Get child IDs
                        child_ids = merge_child_ids.isel({self.timedim: t, 'merge': merge_idx}).compute().values
                        child_ids = child_ids[child_ids >= 0]
                        
                        # Get parent IDs and areas
                        parent_ids = merge_parent_ids.isel({self.timedim: t, 'merge': merge_idx}).compute().values
                        areas = merge_areas.isel({self.timedim: t, 'merge': merge_idx}).compute().values
                        valid_mask = parent_ids >= 0
                        parent_ids = parent_ids[valid_mask]
                        areas = areas[valid_mask]
                        
                        # Map temporary IDs to permanent IDs
                        mapped_child_ids = [id_lookup.get(id_.item(), id_.item()) for id_ in child_ids]
                        mapped_parent_ids = [id_lookup.get(id_.item(), id_.item()) for id_ in parent_ids]
                        
                        # Store in pre-allocated arrays
                        child_ids_iter[t, merge_idx, :len(mapped_child_ids)] = mapped_child_ids
                        parent_ids_iter[t, merge_idx, :len(mapped_parent_ids)] = mapped_parent_ids
                        merge_areas_iter[t, merge_idx, :len(areas)] = areas
            
            # Process final merging objects for next iteration
            final_merging_objects = final_merging_objects.compute().values
            final_merging_objects = final_merging_objects[final_merging_objects > 0]
            mapped_final_objects = [id_lookup.get(id_, id_) for id_ in final_merging_objects]
            new_merging_objects.update(mapped_final_objects)
            
            if self.verbosity > 1:
                print('  Finished Consolidation Step 3: Merge List Dictionary Consolidation')
            
            # Clean up memory
            del merge_child_ids, merge_parent_ids, merge_areas, merge_counts, has_merge
            gc.collect()
                        
            return (
                object_id_field_unique,  
                (child_ids_iter, parent_ids_iter, merge_areas_iter, merge_counts_iter),
                new_merging_objects, 
                global_id_counter
            )
        
        # ============================
        # Main Loop for Parallel Merging
        # ============================
        
        # Find overlapping objects
        overlap_objects_list = self.find_overlapping_objects(object_id_field_unique)  # List object pairs that overlap by at least overlap_threshold percent
        overlap_objects_list = self.enforce_overlap_threshold(overlap_objects_list, object_props)
        if self.verbosity > 0:
            print('Finished finding overlapping objects')
        
        # Find initial merging objects
        unique_children, children_counts = np.unique(overlap_objects_list[:, 1], return_counts=True)
        merging_objects = set(unique_children[children_counts > 1].astype(np.int32))
        del overlap_objects_list
        
        ## Process chunks iteratively until no new merging objects remain
        
        iteration = 0
        processed_chunks = set()
        global_id_counter = object_props.ID.max().item() + 1
        
        # Initialize global merge event tracking
        global_child_ids = []
        global_parent_ids = []
        global_merge_areas = []
        global_merge_tidx = []
        
        while merging_objects and iteration < self.max_iteration:
            object_id_field_new, merge_data_iter, new_merging_objects, global_id_counter = merge_objects_parallel_iteration(
                object_id_field_unique, merging_objects, global_id_counter
            )
            child_ids_iter, parent_ids_iter, merge_areas_iter, merge_counts_iter = merge_data_iter
            
            # Consolidate merge events from this iteration
            for t in range(len(merge_counts_iter)):
                count = merge_counts_iter[t]
                if count > 0:
                    for merge_idx in range(count):
                        # Extract valid children
                        children = child_ids_iter[t, merge_idx]
                        children = children[children >= 0]
                        
                        # Extract valid parents and areas
                        parents = parent_ids_iter[t, merge_idx]
                        areas = merge_areas_iter[t, merge_idx]
                        valid_mask = parents >= 0
                        parents = parents[valid_mask]
                        areas = areas[valid_mask]
                        
                        # Record valid merge events
                        if len(children) > 0 and len(parents) > 0:
                            global_child_ids.append(children)
                            global_parent_ids.append(parents)
                            global_merge_areas.append(areas)
                            global_merge_tidx.append(t)
            
            # Prepare for next iteration - only process objects not already handled
            merging_objects = new_merging_objects - processed_chunks
            processed_chunks.update(new_merging_objects)
            iteration += 1
            
            # Update the object field
            object_id_field_unique = object_id_field_new
            del object_id_field_new
        
        # Check if we reached maximum iterations
        if iteration == self.max_iteration:
            raise RuntimeError(
                f"Reached maximum iterations ({self.max_iteration}) in split_and_merge_objects_parallel. "
                f"Set optional argument 'max_iteration' to a higher value."
            )
        
        ## Process the collected merge events
        
        times = object_id_field_unique[self.timedim].values
        
        # Find maximum dimensions for arrays
        max_parents = max(len(ids) for ids in global_parent_ids)
        max_children = max(len(ids) for ids in global_child_ids)
        
        # Create padded arrays for merge events
        parent_ids_array = np.full((len(global_parent_ids), max_parents), -1, dtype=np.int32)
        child_ids_array = np.full((len(global_child_ids), max_children), -1, dtype=np.int32)
        overlap_areas_array = np.full(
            (len(global_merge_areas), max_parents), 
            -1, 
            dtype=np.float32 if self.unstructured_grid else np.int32
        )
        
        # Fill arrays with merge data
        for i, parents in enumerate(global_parent_ids):
            parent_ids_array[i, :len(parents)] = parents

        for i, children in enumerate(global_child_ids):
            child_ids_array[i, :len(children)] = children

        for i, areas in enumerate(global_merge_areas):
            overlap_areas_array[i, :len(areas)] = areas
        
        # Create merge events dataset
        merge_events = xr.Dataset(
            {
                'parent_IDs': (('merge_ID', 'parent_idx'), parent_ids_array),
                'child_IDs': (('merge_ID', 'child_idx'), child_ids_array),
                'overlap_areas': (('merge_ID', 'parent_idx'), overlap_areas_array),
                'merge_time': ('merge_ID', times[global_merge_tidx]),
                'n_parents': ('merge_ID', np.array([len(p) for p in global_parent_ids], dtype=np.int8)),
                'n_children': ('merge_ID', np.array([len(c) for c in global_child_ids], dtype=np.int8))
            },
            attrs={
                'fill_value': -1
            }
        )
        
        # Recompute object properties and overlaps after all merging
        object_id_field_unique = object_id_field_unique.persist(optimize_graph=True)
        object_props = self.calculate_object_properties(
            object_id_field_unique, 
            properties=['area', 'centroid']
        ).persist(optimize_graph=True)
        
        # Recompute overlaps based on final object configuration
        overlap_objects_list = self.find_overlapping_objects(object_id_field_unique)
        overlap_objects_list = self.enforce_overlap_threshold(overlap_objects_list, object_props)
        overlap_objects_list = overlap_objects_list[:, :2].astype(np.int32)
        
        return (
            object_id_field_unique,
            object_props,
            overlap_objects_list,
            merge_events
        )





"""
MarEx Helper Functions

These are the remaining implementations of helper functions for the MarEx package,
providing optimised algorithms for partitioning, distance calculations, and spatial
operations on both structured and unstructured grids.
"""

@jit(nopython=True, parallel=True, fastmath=True)
def wrapped_euclidian_parallel(mask_values, parent_centroids_values, Nx):
    """
    Optimised function for computing wrapped Euclidean distances.
    
    Efficiently calculates distances between points in a binary mask and a set of
    centroids, accounting for periodic boundaries in the x dimension.
    
    Parameters
    ----------
    mask_values : np.ndarray
        2D boolean array where True indicates points to calculate distances for
    parent_centroids_values : np.ndarray
        Array of shape (n_parents, 2) containing (y, x) coordinates of parent centroids
    Nx : int
        Size of the x-dimension for periodic boundary wrapping
        
    Returns
    -------
    distances : np.ndarray
        Array of shape (n_true_points, n_parents) with minimum distances
    """
    n_parents = len(parent_centroids_values)
    half_Nx = Nx / 2
    
    y_indices, x_indices = np.nonzero(mask_values)
    n_true = len(y_indices)
    
    distances = np.empty((n_true, n_parents), dtype=np.float64)
    
    # Precompute for faster access
    parent_y = parent_centroids_values[:, 0]
    parent_x = parent_centroids_values[:, 1]
    
    # Parallel loop over true positions
    for idx in prange(n_true):
        y, x = y_indices[idx], x_indices[idx]
        
        # Pre-compute y differences for all parents
        dy = y - parent_y
        
        # Pre-compute x differences for all parents
        dx = x - parent_x
        
        # Wrapping correction
        dx = np.where(dx > half_Nx, dx - Nx, dx)
        dx = np.where(dx < -half_Nx, dx + Nx, dx)
        
        distances[idx] = np.sqrt(dy * dy + dx * dx)
    
    return distances

@jit(nopython=True, fastmath=True)
def create_grid_index_arrays(points_y, points_x, grid_size, ny, nx):
    """
    Create a grid-based spatial index for efficient point lookup.
    
    This function divides space into a grid and assigns points to grid cells
    for more efficient spatial queries compared to brute force comparisons.
    
    Parameters
    ----------
    points_y, points_x : np.ndarray
        Coordinates of points to index
    grid_size : int
        Size of each grid cell
    ny, nx : int
        Dimensions of the overall grid
        
    Returns
    -------
    grid_points : np.ndarray
        3D array mapping grid cells to point indices
    grid_counts : np.ndarray
        2D array with count of points in each grid cell
    """
    n_grids_y = (ny + grid_size - 1) // grid_size
    n_grids_x = (nx + grid_size - 1) // grid_size
    max_points_per_cell = len(points_y)
    
    grid_points = np.full((n_grids_y, n_grids_x, max_points_per_cell), -1, dtype=np.int32)
    grid_counts = np.zeros((n_grids_y, n_grids_x), dtype=np.int32)
    
    for idx in range(len(points_y)):
        grid_y = min(points_y[idx] // grid_size, n_grids_y - 1)
        grid_x = min(points_x[idx] // grid_size, n_grids_x - 1)
        count = grid_counts[grid_y, grid_x]
        if count < max_points_per_cell:
            grid_points[grid_y, grid_x, count] = idx
            grid_counts[grid_y, grid_x] += 1
    
    return grid_points, grid_counts

@jit(nopython=True, fastmath=True)
def calculate_wrapped_distance(y1, x1, y2, x2, nx, half_nx):
    """
    Calculate distance with periodic boundary conditions in x dimension.
    
    Parameters
    ----------
    y1, x1 : float
        Coordinates of first point
    y2, x2 : float
        Coordinates of second point
    nx : int
        Size of x dimension
    half_nx : float
        Half the size of x dimension
        
    Returns
    -------
    float
        Euclidean distance accounting for periodic boundary in x
    """
    dy = y1 - y2
    dx = x1 - x2
    
    if dx > half_nx:
        dx -= nx
    elif dx < -half_nx:
        dx += nx
        
    return np.sqrt(dy * dy + dx * dx)

@jit(nopython=True, parallel=True, fastmath=True)
def partition_nn_grid(child_mask, parent_masks, child_ids, parent_centroids, Nx, max_distance=20):
    """
    Partition a child object based on nearest parent object points.
    
    This implementation uses spatial indexing and highly-threaded processing 
    for efficient distance calculations. The algorithm assigns each point
    in the child object to the closest parent object.
    
    Parameters
    ----------
    child_mask : np.ndarray
        Binary mask of the child object
    parent_masks : np.ndarray
        List of binary masks for each parent object
    child_ids : np.ndarray
        List of IDs to assign to partitions
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) with parent centroids
    Nx : int
        Size of x dimension for periodic boundaries
    max_distance : int, default=20
        Maximum search distance
        
    Returns
    -------
    new_labels : np.ndarray
        Array containing assigned child_ids for each point
    """
    ny, nx = child_mask.shape
    half_Nx = Nx / 2
    n_parents = len(parent_masks)
    grid_size = max(2, max_distance // 4)
    
    y_indices, x_indices = np.nonzero(child_mask)
    n_child_points = len(y_indices)
    
    min_distances = np.full(n_child_points, np.inf)
    parent_assignments = np.zeros(n_child_points, dtype=np.int32)
    found_close = np.zeros(n_child_points, dtype=np.bool_)
    
    for parent_idx in range(n_parents):
        py, px = np.nonzero(parent_masks[parent_idx])
        
        if len(py) == 0:  # Skip empty parents
            continue
            
        # Create grid index for this parent
        n_grids_y = (ny + grid_size - 1) // grid_size
        n_grids_x = (nx + grid_size - 1) // grid_size
        grid_points, grid_counts = create_grid_index_arrays(py, px, grid_size, ny, nx)
        
        # Process child points in parallel
        for child_idx in prange(n_child_points):
            if found_close[child_idx]:  # Skip if we already found an exact match
                continue
                
            child_y, child_x = y_indices[child_idx], x_indices[child_idx]
            grid_y = min(child_y // grid_size, n_grids_y - 1)
            grid_x = min(child_x // grid_size, n_grids_x - 1)
            
            min_dist_to_parent = np.inf
            
            # Check nearby grid cells
            for dy in range(-1, 2):
                grid_y_check = (grid_y + dy) % n_grids_y
                
                for dx in range(-1, 2):
                    grid_x_check = (grid_x + dx) % n_grids_x
                    
                    # Process points in this grid cell
                    n_points = grid_counts[grid_y_check, grid_x_check]
                    
                    for p_idx in range(n_points):
                        point_idx = grid_points[grid_y_check, grid_x_check, p_idx]
                        if point_idx == -1:
                            break
                        
                        dist = calculate_wrapped_distance(
                            child_y, child_x,
                            py[point_idx], px[point_idx],
                            Nx, half_Nx
                        )
                        
                        if dist > max_distance:
                            continue
                        
                        if dist < min_dist_to_parent:
                            min_dist_to_parent = dist
                            
                        if dist < 1e-6:  # Found exact same point (within numerical precision)
                            min_dist_to_parent = dist
                            found_close[child_idx] = True
                            break
                    
                    if found_close[child_idx]:
                        break
                
                if found_close[child_idx]:
                    break
            
            # Update assignment if this parent is closer
            if min_dist_to_parent < min_distances[child_idx]:
                min_distances[child_idx] = min_dist_to_parent
                parent_assignments[child_idx] = parent_idx
    
    # Handle any unassigned points using centroids
    unassigned = min_distances == np.inf
    if np.any(unassigned):
        for child_idx in np.nonzero(unassigned)[0]:
            child_y, child_x = y_indices[child_idx], x_indices[child_idx]
            min_dist = np.inf
            best_parent = 0
            
            for parent_idx in range(n_parents):
                # Calculate distance to centroid with periodic boundary conditions
                dist = calculate_wrapped_distance(
                    child_y, child_x,
                    parent_centroids[parent_idx, 0],
                    parent_centroids[parent_idx, 1],
                    Nx, half_Nx
                )
                
                if dist < min_dist:
                    min_dist = dist
                    best_parent = parent_idx
                    
            parent_assignments[child_idx] = best_parent
    
    # Convert from parent indices to child_ids
    new_labels = child_ids[parent_assignments]
    
    return new_labels

@jit(nopython=True, fastmath=True)
def partition_nn_unstructured(child_mask, parent_masks, child_ids, parent_centroids, neighbours_int, lat, lon, max_distance=20):
    """
    Partition a child object on an unstructured grid based on nearest parent points.
    
    This function implements an efficient algorithm for assigning each cell in a child
    object to the nearest parent object, using graph traversal and spatial distances.
    It is optimised for unstructured grids.
    
    Parameters
    ----------
    child_mask : np.ndarray
        1D boolean array where True indicates points in the child object
    parent_masks : np.ndarray
        2D boolean array of shape (n_parents, n_points) where True indicates points in each parent object
    child_ids : np.ndarray
        1D array containing the IDs to assign to each partition of the child object
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) containing (lat, lon) coordinates of parent centroids in degrees
    neighbours_int : np.ndarray
        2D array of shape (3, n_points) containing indices of neighboring cells for each point
    lat, lon : np.ndarray
        Latitude/longitude arrays in degrees
    max_distance : int, default=20
        Maximum number of edge hops to search for parent points
    
    Returns
    -------
    new_labels : np.ndarray
        1D array containing the assigned child_ids for each True point in child_mask
    """
    
    # Force contiguous arrays in memory for optimal vectorised performance
    child_mask = np.ascontiguousarray(child_mask)
    parent_masks = np.ascontiguousarray(parent_masks)
    
    n_points = len(child_mask)
    n_parents = len(parent_masks)
    
    # Pre-allocate arrays
    distances = np.full(n_points, np.inf, dtype=np.float32)
    parent_assignments = np.full(n_points, -1, dtype=np.int32)
    visited = np.zeros((n_parents, n_points), dtype=np.bool_)
    
    # Initialise with direct overlaps
    for parent_idx in range(n_parents):
        overlap_mask = parent_masks[parent_idx] & child_mask
        if np.any(overlap_mask):
            visited[parent_idx, overlap_mask] = True
            unclaimed_overlap = distances[overlap_mask] == np.inf
            if np.any(unclaimed_overlap):
                overlap_points = np.where(overlap_mask)[0]
                valid_points = overlap_points[unclaimed_overlap]
                distances[valid_points] = 0
                parent_assignments[valid_points] = parent_idx
    
    # Pre-compute trig values for efficiency
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    cos_lat = np.cos(lat_rad)
    
    # Graph traversal for remaining points - expanding from parent frontiers
    current_distance = 0
    any_unassigned = np.any(child_mask & (parent_assignments == -1))
    
    while current_distance < max_distance and any_unassigned:
        current_distance += 1
        updates_made = False
        
        for parent_idx in range(n_parents):
            # Get current frontier points
            frontier_mask = visited[parent_idx]
            if not np.any(frontier_mask):
                continue
            
            # Process neighbors
            for i in range(3):  # For each neighbor direction
                neighbors = neighbours_int[i, frontier_mask]
                valid_neighbors = neighbors >= 0
                if not np.any(valid_neighbors):
                    continue
                    
                valid_points = neighbors[valid_neighbors]
                unvisited = ~visited[parent_idx, valid_points]
                new_points = valid_points[unvisited]
                
                if len(new_points) > 0:
                    visited[parent_idx, new_points] = True
                    update_mask = distances[new_points] > current_distance
                    if np.any(update_mask):
                        points_to_update = new_points[update_mask]
                        distances[points_to_update] = current_distance
                        parent_assignments[points_to_update] = parent_idx
                        updates_made = True
        
        if not updates_made:
            break
            
        any_unassigned = np.any(child_mask & (parent_assignments == -1))
    
    # Handle remaining unassigned points using great circle distances
    unassigned_mask = child_mask & (parent_assignments == -1)
    if np.any(unassigned_mask):
        parent_lat_rad = np.deg2rad(parent_centroids[:, 0])
        parent_lon_rad = np.deg2rad(parent_centroids[:, 1])
        cos_parent_lat = np.cos(parent_lat_rad)
        
        unassigned_points = np.where(unassigned_mask)[0]
        for point in unassigned_points:
            # Vectorized haversine calculation
            dlat = parent_lat_rad - lat_rad[point]
            dlon = parent_lon_rad - lon_rad[point]
            a = np.sin(dlat/2)**2 + cos_lat[point] * cos_parent_lat * np.sin(dlon/2)**2
            dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            parent_assignments[point] = np.argmin(dist)
    
    # Return only the assignments for points in child_mask
    child_points = np.where(child_mask)[0]
    return child_ids[parent_assignments[child_points]]

@jit(nopython=True, fastmath=True)
def partition_nn_unstructured_optimised(child_mask, parent_frontiers, parent_centroids, neighbours_int, lat, lon, max_distance=20):
    """
    Memory-optimised nearest neighbor partitioning for unstructured grids.
    
    This version uses more efficient memory management compared to partition_nn_unstructured,
    making it suitable for very large grids. It uses a compact representation of parent
    frontiers to reduce memory usage during graph traversal.
    
    Parameters
    ----------
    child_mask : np.ndarray
        1D boolean array indicating which cells belong to the child object
    parent_frontiers : np.ndarray
        1D uint8 array with parent indices (255 for unvisited points)
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) containing (lat, lon) coordinates
    neighbours_int : np.ndarray
        2D array of shape (3, n_points) containing indices of neighboring cells
    lat, lon : np.ndarray
        1D arrays of latitude/longitude in degrees
    max_distance : int, default=20
        Maximum number of edge hops to search for parent points
    
    Returns
    -------
    result : np.ndarray
        1D array containing the assigned parent indices for points in child_mask
    """
    
    # Create working copies to ensure memory cleanup
    parent_frontiers_working = parent_frontiers.copy()
    child_mask_working = child_mask.copy()
    
    n_parents = np.max(parent_frontiers_working[parent_frontiers_working < 255]) + 1
    
    # Graph traversal - expanding frontiers
    current_distance = 0
    any_unassigned = np.any(child_mask_working & (parent_frontiers_working == 255))
    
    while current_distance < max_distance and any_unassigned:
        current_distance += 1
        updates_made = False
        
        for parent_idx in range(n_parents):
            # Skip if no frontier points for this parent
            if not np.any(parent_frontiers_working == parent_idx):
                continue
            
            # Process neighbours for current parent's frontier
            for i in range(3):
                neighbors = neighbours_int[i, parent_frontiers_working == parent_idx]
                valid_neighbors = neighbors >= 0
                
                if not np.any(valid_neighbors):
                    continue
                
                valid_points = neighbors[valid_neighbors]
                unvisited = parent_frontiers_working[valid_points] == 255
                
                if not np.any(unvisited):
                    continue
                
                # Update new frontier points
                new_points = valid_points[unvisited]
                parent_frontiers_working[new_points] = parent_idx
                
                if np.any(child_mask_working[new_points]):
                    updates_made = True
        
        if not updates_made:
            break
            
        any_unassigned = np.any(child_mask_working & (parent_frontiers_working == 255))
    
    # Handle remaining unassigned points using great circle distances
    unassigned_mask = child_mask_working & (parent_frontiers_working == 255)
    if np.any(unassigned_mask):
        # Pre-compute parent coordinates in radians
        parent_lat_rad = np.deg2rad(parent_centroids[:, 0])
        parent_lon_rad = np.deg2rad(parent_centroids[:, 1])
        cos_parent_lat = np.cos(parent_lat_rad)
        
        # Process each unassigned point
        unassigned_points = np.where(unassigned_mask)[0]
        for point in unassigned_points:
            dlat = parent_lat_rad - np.deg2rad(lat[point])
            dlon = parent_lon_rad - np.deg2rad(lon[point])
            
            a = np.sin(dlat/2)**2 + np.cos(np.deg2rad(lat[point])) * cos_parent_lat * np.sin(dlon/2)**2
            dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            parent_frontiers_working[point] = np.argmin(dist)
    
    # Extract result for child points only
    result = parent_frontiers_working[child_mask_working].copy()
    
    # Explicitly clear working arrays to help with memory management
    parent_frontiers_working = None
    child_mask_working = None
    
    return result

@jit(nopython=True, parallel=True, fastmath=True)
def partition_centroid_unstructured(child_mask, parent_centroids, child_ids, lat, lon):
    """
    Partition a child object based on closest parent centroids on an unstructured grid.
    
    This function assigns each cell in the child object to the parent with the closest
    centroid, using great circle distances on a spherical grid.
    
    Parameters
    ----------
    child_mask : np.ndarray
        1D boolean array indicating which cells belong to the child object
    parent_centroids : np.ndarray
        Array of shape (n_parents, 2) containing (lat, lon) coordinates of parent centroids in degrees
    child_ids : np.ndarray
        Array of IDs to assign to each partition of the child object
    lat, lon : np.ndarray
        Latitude/longitude arrays in degrees
        
    Returns
    -------
    new_labels : np.ndarray
        1D array containing assigned child_ids for cells in child_mask
    """
    n_cells = len(child_mask)
    n_parents = len(parent_centroids)
    
    # Convert to radians for spherical calculations
    lat_rad = np.deg2rad(lat)
    lon_rad = np.deg2rad(lon)
    parent_coords_rad = np.deg2rad(parent_centroids)
    
    new_labels = np.zeros(n_cells, dtype=child_ids.dtype)
    
    # Process each child cell in parallel
    for i in prange(n_cells):
        if not child_mask[i]:
            continue
            
        min_dist = np.inf
        closest_parent = 0
        
        # Calculate great circle distance to each parent centroid
        for j in range(n_parents):
            dlat = parent_coords_rad[j, 0] - lat_rad[i]
            dlon = parent_coords_rad[j, 1] - lon_rad[i]
            
            # Use haversine formula for great circle distance
            a = np.sin(dlat/2)**2 + np.cos(lat_rad[i]) * np.cos(parent_coords_rad[j, 0]) * np.sin(dlon/2)**2
            dist = 2 * np.arctan2(np.sqrt(a), np.sqrt(1-a))
            
            if dist < min_dist:
                min_dist = dist
                closest_parent = j
        
        new_labels[i] = child_ids[closest_parent]
    
    return new_labels

@njit(fastmath=True, parallel=True)
def sparse_bool_power(vec, sp_data, indices, indptr, exponent):
    """
    Efficient sparse boolean matrix power operation.
    
    This function implements a fast sparse matrix power operation for boolean matrices,
    avoiding memory leaks present in scipy+Dask implementations. It's used for efficient
    morphological operations on unstructured grids.
    
    Parameters
    ----------
    vec : np.ndarray
        Boolean vector to multiply
    sp_data, indices, indptr : np.ndarray
        Sparse matrix in CSR format
    exponent : int
        Number of times to apply the matrix
        
    Returns
    -------
    np.ndarray
        Result of (sparse_matrix ^ exponent) * vec
    """
    vec = vec.T
    num_rows = indptr.size - 1
    num_cols = vec.shape[1]
    result = vec.copy()

    for _ in range(exponent):
        temp_result = np.zeros((num_rows, num_cols), dtype=np.bool_)

        for i in prange(num_rows):
            for j in range(indptr[i], indptr[i + 1]):
                if sp_data[j]:
                    for k in range(num_cols):
                        if result[indices[j], k]:
                            temp_result[i, k] = True

        result = temp_result

    return result.T