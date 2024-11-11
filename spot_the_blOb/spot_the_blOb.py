import xarray as xr
import numpy as np
import scipy.ndimage
from skimage.measure import regionprops_table 
from dask_image.ndmeasure import label
import dask.array as dsa
from dask import persist
from dask.base import is_dask_collection

class Spotter:
        
    def __init__(self, data_bin, mask, R_fill, area_filter_quartile, allow_merging=True, timedim='time', xdim='lat', ydim='lon'):
        
        self.data_bin           = data_bin
        self.mask               = mask
        self.R_fill             = int(R_fill)
        self.area_filter_quartile   = area_filter_quartile
        self.allow_merging      = allow_merging
        self.timedim    = timedim
        self.xdim       = xdim
        self.ydim       = ydim   
        
        if ((timedim, ydim, xdim) != data_bin.dims):
            try:
                data_bin = data_bin.transpose(timedim, ydim, xdim) 
            except:
                raise ValueError(f'Spot_the_blOb currently only supports 3D DataArrays. The dimensions should only contain ({timedim}, {xdim}, and {ydim}). Found {list(data_bin.dims)}')

        if (data_bin.data.dtype != np.bool):
            raise ValueError('The input DataArray is not binary. Please convert to a binary array, and try again.  :)')
        
        if not is_dask_collection(data_bin.data):
            raise ValueError('The input DataArray is not backed by a Dask array. Please chunk (in time), and try again.  :)')
        
        if (mask.data.dtype != np.bool):
            raise ValueError('The mask not binary. Please convert to a binary array, and try again.  :)')
        
        if (mask == False).all():
            raise ValueError('Found only False in `mask` input. The mask should indicate valid regions with True values.')
        
        if (area_filter_quartile < 0) or (area_filter_quartile > 1):
            raise ValueError('The discard_fraction should be between 0 and 1.')
        
            
    def run(self):
        '''
        Cluster, label, filter, and track objects in a binary field with optional merging & splitting. 
        
        Parameters
        ----------
        data_bin : xarray.DataArray
            The _binary_ data to group & label. Must represent an underlying dask array.

        mask : xarray.DataArray
            The _binary_ mask of points to keep. False indicates points to ignore. 

        R_fill : int
            The size of the structuring element used in morphological opening & closing, relating to the largest hole that can be filled. In units of pixels.

        discard_fraction : float
            The fraction of the smallest objects to discard, i.e. the quantile defining the smallest area object retained. Value should be between 0 and 1.
        
        Returns
        -------
        blob_id_field : xarray.DataArray
            Field of globally unique integer labels of each element in connected regions. ID = 0 indicates no object.
        '''

        # Fill Small Holes & Gaps between Objects
        data_bin_filled = self.fill_holes()

        # Remove Small Objects
        data_bin_filtered, area_threshold, blob_areas, N_blobs_unfiltered = self.filter_small_blobs(data_bin_filled)

        if not self.allow_merging:
            # Track Blobs without any special merging or splitting
            blob_id_field, N_blobs_final = self.identify_blobs(data_bin_filtered, time_connectivity=True)
        else:
            # Track Blobs with merging & splitting
            blob_id_field, N_blobs_final = self.track_blObs(data_bin_filtered)


        ## Save Some BlObby Stats

        total_id_area = int(blob_areas.sum().item())  # Percent of total object area retained after size filtering
        
        rejected_area = blob_areas.where(blob_areas <= area_threshold, drop=True).sum().item()
        rejected_area_fraction = rejected_area / total_id_area

        accepted_area = blob_areas.where(blob_areas > area_threshold, drop=True).sum().item()
        accepted_area_fraction = accepted_area / total_id_area

        blob_id_field = blob_id_field.rename('labels')
        blob_id_field.attrs['N_blobs_unfiltered'] = int(N_blobs_unfiltered)
        blob_id_field.attrs['N_blobs_final'] = int(N_blobs_final)
        blob_id_field.attrs['R_fill'] = self.R_fill
        blob_id_field.attrs['area_filter_quartile'] = self.area_filter_quartile
        blob_id_field.attrs['area_threshold'] = area_threshold
        blob_id_field.attrs['rejected_area_fraction'] = rejected_area_fraction
        blob_id_field.attrs['accepted_area_fraction'] = accepted_area_fraction

        ## Print Some BlObby Stats
        
        print(f'Total Object Area: {total_id_area}')
        print(f'Number of Initial Blobs: {N_blobs_unfiltered}')
        print(f'Area Cutoff Threshold: {area_threshold}')
        print(f'Rejected Area Fraction: {rejected_area_fraction}')
        print(f'Total Blobs Tracked: {N_blobs_final}')
        
        return blob_id_field
    
    

    def fill_holes(self): 
        '''
        Performs morphological closing then opening to fill in gaps & holes up to size R_fill.
        
        Parameters
        ----------
        R_fill : int
            Length of grid spacing to define the size of the structing element used in morphological closing and opening.
        
        Returns
        -------
        data_bin_filled_mask : xarray.DataArray
            Binary data with holes/gaps filled and masked.
        '''
        
        # Generate Structuring Element
        diameter = 2 * self.R_fill
        x = np.arange(-self.R_fill, self.R_fill+1)
        x, y = np.meshgrid(x, x)
        r = x**2 + y**2 
        se_kernel = r < (self.R_fill**2)+1

        def binary_open_close(binary_data):
            binary_data_padded = np.pad(binary_data,
                                          ((diameter, diameter), (diameter, diameter)),
                                          mode='wrap')
            binary_data_closed = scipy.ndimage.binary_closing(binary_data_padded, se_kernel)
            binary_data_opened = scipy.ndimage.binary_opening(binary_data_closed, se_kernel, output=binary_data_padded)
            return binary_data_opened[diameter:-diameter, diameter:-diameter]

        data_bin_filled = xr.apply_ufunc(binary_open_close, self.data_bin,
                                   input_core_dims=[[self.ydim, self.xdim]],
                                   output_core_dims=[[self.ydim, self.xdim]],
                                   output_dtypes=[self.data_bin.dtype],
                                   vectorize=True,
                                   dask='parallelized')
        
        # Mask out edge features arising from Morphological Operations
        data_bin_filled_mask = data_bin_filled.where(self.mask, drop=False, other=False)
        
        return data_bin_filled_mask


    def identify_blobs(self, data_bin, time_connectivity):
        '''Labels connected regions in the binary data.
        
        Returns
        -------
        blob_id_field : xarray.DataArray
            Field of integer labels of each element in connected regions. ID = 0 indicates no object.
        '''
        
        neighbours = np.zeros((3,3,3))
        neighbours[1,:,:] = 1           # Connectivity Kernel: All 8 neighbours, but ignore time
        
        if time_connectivity:
            # ID blobs in 3D (i.e. space & time) -- N.B. Labels are unique across time
            neighbours[:,1,1] = 1 #                         including +-1 in time
        # else, ID blobs only in 2D (i.e. space) -- N.B. Labels are _not_ unique across time     
        
        # Cluster & Label Binary Data
        blob_id_field, N_blobs = persist(label(data_bin,           # Apply dask-powered ndimage & persist in memory
                                            structure=neighbours, 
                                            wrap_axes=(2,)))       # Wrap in x-direction
        
        N_blobs = N_blobs.compute()
        # DataArray (same shape as data_bin) but with integer labels: 
        blob_id_field = xr.DataArray(blob_id_field, coords=data_bin.coords, dims=data_bin.dims, attrs=data_bin.attrs)
        
        return blob_id_field, N_blobs


    def filter_small_blobs(self, data_bin):
        '''Filters out smallest ojects in the binary data.'''
        
        ## Cluster & Label Binary Data: Time-independent in 2D (i.e. no time connectivity!)
        blob_id_field, N_blobs = self.identify_blobs(data_bin, time_connectivity=False)
        
        ## Calculate Area of each Blob
        def compute_blob_areas(ids):
            props_slice = regionprops_table(ids, properties=['label', 'area'])
            return props_slice
        
        blob_props = xr.apply_ufunc(compute_blob_areas, blob_id_field,
                                    input_core_dims=[[self.ydim, self.xdim]],
                                    output_core_dims=[[]],
                                    output_dtypes=[object],
                                    vectorize=True,
                                    dask='parallelized')
        
        # Convert to an xarray DataSet
        blob_props = xr.concat([xr.Dataset({key: (['label'], value) for key, value in item.items()}) for item in blob_props.values], dim='label') 
        blob_areas = blob_props.area
        blob_ids = blob_props.label
        
        if blob_areas.size == 0:
            raise ValueError(f'No objects were detected.')
        
        ## Remove Smallest Blobs
        area_threshold = np.percentile(blob_areas, self.area_filter_quartile*100.0)
        blob_ids_keep = blob_ids.where(blob_areas >= area_threshold, drop=True)
        data_bin_filtered = blob_id_field.isin(blob_ids_keep)

        return data_bin_filtered, area_threshold, blob_areas, N_blobs
    
    
    def track_blObs(self, data_bin):
        '''Identifies & Labels Blobs across time, accounting for splitting & merging logic.
        
        Returns
        -------
        blob_id_field : xarray.DataArray
            Field of globally unique integer labels of each element in connected regions. ID = 0 indicates no object.
        '''
        
        ## Cluster & Label Binary Data at each Time Step
        blob_id_field, _ = self.identify_blobs(data_bin, time_connectivity=False)
        
        ## Generate Unique Blob IDs
        #  ... add cumulative blob id to each next time step...
        
        ## Calculate Area of each Blob
        #  ... regionprops_table()...
        
        ## Compile List of Overlapping Blob ID Pairs Across Time
        overlap_pairs # Look +- 1 in time...
        #  ... np.unique()...
        
        ## Apply Splitting & Merging Logic to `overlap_pairs
        equivalent_labels   # Array of lists containing all equivalent labels (i.e. blobs that are the same object either through space or time)
        
        ## Relabel Blobs with Unique IDs
        blob_id_field
        
        ## Count Number of Blobs (This may have increased due to splitting)
        N_blobs
    
        return blob_id_field, N_blobs