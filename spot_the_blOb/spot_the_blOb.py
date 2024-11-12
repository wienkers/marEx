import xarray as xr
import numpy as np
from dask_image.ndmeasure import label
from skimage.measure import regionprops_table
from dask_image.ndmorph import binary_closing, binary_opening
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
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

        if (self.data_bin[xdim].max() < 10.0):
            print('The lat/lon coordinates appear to be in degrees. Converting to radians...')
            self.data_bin[xdim] = np.deg2rad(self.data_bin[xdim])
            self.data_bin[ydim] = np.deg2rad(self.data_bin[ydim])
            self.converted_to_radians = True
        else:
            self.converted_to_radians = False
        
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
            # Track Blobs _with_ Merging & Splitting
            blob_id_field, N_blobs_final = self.track_blObs(data_bin_filtered)

        if self.converted_to_radians: # Convert back to degrees
            blob_id_field[self.xdim] = np.rad2deg(blob_id_field[self.xdim])
            blob_id_field[self.ydim] = np.rad2deg(blob_id_field[self.ydim])
        
        
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
        
        binary_data_padded = self.data_bin.pad({self.ydim: diameter, self.xdim: diameter, }, mode='wrap')
        binary_data_closed = binary_closing(binary_data_padded, structure=se_kernel)
        binary_data_opened = binary_opening(binary_data_closed, structure=se_kernel)
        data_bin_filled    = binary_data_opened.isel({self.ydim: slice(diameter, -diameter), self.xdim: slice(diameter, -diameter)})
        
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
    
    
    def calculate_centroid(binary_mask, original_centroid):
        '''Calculates the centroid of the binary data when blob touches x-dimension edges.
        
        Parameters:
        -----------
        binary_mask : numpy.ndarray
            2D binary array where True/1 indicates the blob
        original_centroid : tuple
            (y_centroid, x_centroid) from regionprops_table
            
        Returns:
        --------
        tuple
            (y_centroid, x_centroid)
        '''
        
        ..........
        
        # Check if blob touches either edge of x dimension
        x_indices = np.nonzero(binary_mask)[1]  # Only need x coordinates
        touches_left = np.any(x_indices == 0)
        touches_right = np.any(x_indices == binary_mask.shape[1] - 1)
        
        # If blob touches both edges, recalculate x centroid
        if touches_left and touches_right:
            # Adjust x coordinates that are near right edge
            x_indices_adj = x_indices.copy()
            right_side = x_indices > binary_mask.shape[1] // 2
            x_indices_adj[right_side] -= binary_mask.shape[1]
            
            # Calculate x centroid
            x_centroid = np.mean(x_indices_adj)
            if x_centroid < 0:
                x_centroid += binary_mask.shape[1]
                
            return (original_centroid[0], x_centroid)
        
        # If blob doesn't touch both edges, return original centroid
        return original_centroid
        
    
    
    def calculate_blob_properties(self, blob_id_field, properties=None):
        '''
        Calculates properties of the blobs from the blob_id_field.
        
        Parameters:
        -----------
        blob_id_field : xarray.DataArray
            Field containing blob IDs
        properties : list, optional
            List of properties to calculate. If None, defaults to ['label', 'area'].
            See skimage.measure.regionprops for available properties.
            
        Returns:
        --------
        xarray.Dataset
            Dataset containing all calculated properties with 'label' dimension
        '''
        
        # Default Properties
        if properties is None:
            properties = ['label', 'area']
        
        # 'label' is needed for ID
        if 'label' not in properties:
            properties = ['label'] + properties
        
        check_centroids = 'centroid' in properties
        
        # Define wrapper function to run in parallel
        def blob_properties_chunk(ids):
            
            # Calculate Standard Properties
            props_slice = regionprops_table(ids, properties=properties)
            
            # Check Centroids if blob touches either edge (Need to account for x-dimension edge wrapping)
            if check_centroids and len(props_slice['label']) > 0:
                # Get original centroids
                centroids = list(zip(props_slice['centroid-0'], props_slice['centroid-1']))
                centroids_wrapped = []
                
                # Process each blob
                for label_idx, label in enumerate(props_slice['label']):
                    binary_mask = ids == label
                    centroids_wrapped.append(
                        calculate_centroid(binary_mask, centroids[label_idx])
                    )
                
                # Update centroid values
                props_slice['centroid-0'] = [c[0] for c in centroids_wrapped]
                props_slice['centroid-1'] = [c[1] for c in centroids_wrapped]
            
            return props_slice
        
        
        blob_props = xr.apply_ufunc(blob_properties_chunk, blob_id_field,
                                    input_core_dims=[[self.ydim, self.xdim]],
                                    output_core_dims=[[]],
                                    output_dtypes=[object],
                                    vectorize=True,
                                    dask='parallelized')
        
        # Concatenate and Convert to an xarray Dataset
        blob_props = xr.concat([
            xr.Dataset({key: (['label'], value) for key, value in item.items()}) 
            for item in blob_props.values
        ], dim='label')
        
        if blob_props.label.size == 0:
            raise ValueError(f'No objects were detected.')
        
        return blob_props
    

    def filter_small_blobs(self, data_bin):
        '''Filters out smallest ojects in the binary data.'''
        
        # Cluster & Label Binary Data: Time-independent in 2D (i.e. no time connectivity!)
        blob_id_field, N_blobs = self.identify_blobs(data_bin, time_connectivity=False)
        
        # Compute Blob Areas
        blob_props = self.calculate_blob_properties(blob_id_field)
        blob_areas, blob_ids = blob_props.area, blob_props.label
        
        # Remove Smallest Blobs
        area_threshold = np.percentile(blob_areas, self.area_filter_quartile*100.0)
        blob_ids_keep = blob_ids.where(blob_areas >= area_threshold, drop=True)
        data_bin_filtered = blob_id_field.isin(blob_ids_keep)

        return data_bin_filtered, area_threshold, blob_areas, N_blobs
    
    
    def find_overlapping_blobs(self, blob_id_field):
        '''Finds overlapping blobs across time.'''
        
        ## Check just for overlap with previous and next time slice.
        #  Keep a running list of all equivalent labels.
        
        blob_id_field_prev = blob_id_field.roll(time=1,  roll_coords=False)
        blob_id_field_next = blob_id_field.roll(time=-1, roll_coords=False)
        
        def check_overlap_slice(ids_t0, ids_prev, ids_next):
            '''Finds overlapping blobs in a single time slice by looking at +-1 in time.'''
            
            # Create arrays of indices for valid labels
            id_mask_t0 = ids_t0 > 0
            id_indices_t0 = np.nonzero(id_mask_t0)[0]
            id_indices_prev = id_indices_t0[ids_prev[id_mask_t0] > 0]
            id_indices_next = id_indices_t0[ids_next[id_mask_t0] > 0]

            # Create Pairs using advanced indexing & Concatenate. N.B. We keep the earlier label in time first
            id_pairs = np.concatenate((
                                    np.stack((ids_prev[id_indices_prev], ids_t0[id_indices_prev]),   axis=1),
                                    np.stack((ids_t0[id_indices_next],   ids_next[id_indices_next]), axis=1)), axis=0)
            
            # Find unique pairs
            unique_id_pairs = np.unique(id_pairs, axis=0)

            return unique_id_pairs
        
        # ID Overlapping Blobs in Parallel
        overlap_blobs_list = xr.apply_ufunc(
                            check_overlap_slice,
                            blob_id_field,
                            blob_id_field_prev,
                            blob_id_field_next,
                            input_core_dims=[[self.ydim, self.xdim], [self.ydim, self.xdim], [self.ydim, self.xdim]],
                            output_core_dims=[[]],
                            vectorize=True,
                            dask="parallelized",
                            output_dtypes=[object]
                        ).compute()
        
        overlap_blobs_list_unique = np.unique(np.concatenate(overlap_blobs_list.values), axis=0)
        
        return overlap_blobs_list_unique
    
        
    def cluster_rename_blobs(self, blob_id_field_unique, overlap_blobs_list):
        '''Cluster the blob pairs to determine the final labels, and relabel the blobs.
        
        Parameters
        ----------
        blob_id_field_unique : xarray.DataArray
            Field of unique blob IDs. IDs must not be repeated across time.
        overlap_blobs_list : (N x 2) np.ndarray
            Array of Blob IDs that indicate which blobs are the same. The blob in the first column precedes the second column in time.
        
        Returns
        -------
        split_merged_relabeled_blob_id_field : xarray.DataArray
            Field of renamed blob IDs which track & ID blobs across time. ID = 0 indicates no object.
        '''
        
        ## Cluster the overlap_pairs into groups of equivalent labels
        
        # Get unique IDs from the overlap pairs
        IDs = np.unique(overlap_blobs_list) # 1D sorted unique
        
        # Create a mapping from ID to indices
        ID_to_index = {label: index for index, label in enumerate(IDs)}
        
        # Convert overlap pairs to indices
        overlap_pairs_indices = np.array([(ID_to_index[pair[0]], ID_to_index[pair[1]]) for pair in overlap_blobs_list])
        
        # Create a sparse matrix representation of the graph
        n = len(IDs)
        row_indices, col_indices = overlap_pairs_indices.T
        data = np.ones(len(overlap_pairs_indices))
        graph = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        
        # Solve the graph to determine connected components
        num_components, component_IDs = connected_components(csgraph=graph, directed=False, return_labels=True)
        
        # Group IDs by their component label
        ID_clusters = [[] for _ in range(num_components)]
        for ID, component_ID in zip(IDs, component_IDs):
            ID_clusters[component_ID].append(ID)
        
        
        ## ID_clusters now is a list of lists of equivalent blob IDs that have been tracked across time
        #  We now need to replace all IDs in blob_id_field_unique that match the equivalent_labels with the list index:  This is the new/final ID field.
        
        # Create a dictionary to map labels to the new cluster indices
        min_int32 = np.iinfo(np.int32).min
        max_ID = blob_id_field_unique.max().compute().data
        ID_to_cluster_index_array = np.full(max_ID + 1, min_int32, dtype=np.int32)

        # Fill the lookup array with cluster indices
        for index, cluster in enumerate(ID_clusters):
            for ID in cluster:
                ID_to_cluster_index_array[ID] = np.int32(index) # Because these are the connected labels, there are many fewer!
        
        # N.B.: **Need to pass da into apply_ufunc, otherwise it doesn't manage the memory correctly with large shared-mem numpy arrays**
        ID_to_cluster_index_da = xr.DataArray(ID_to_cluster_index_array, dims='ID', coords={'ID': np.arange(max_ID + 1)})
        
        def map_labels_to_indices(block, ID_to_cluster_index_array):
            mask = block >= 0
            new_block = np.zeros_like(block, dtype=np.int32)
            new_block[mask] = ID_to_cluster_index_array[block[mask]]
            new_block[~mask] = -10
            return new_block
        
        split_merged_relabeled_blob_id_field = xr.apply_ufunc(
            map_labels_to_indices,
            blob_id_field_unique, 
            ID_to_cluster_index_da,
            input_core_dims=[[self.ydim, self.xdim],['ID']],
            output_core_dims=[[self.ydim, self.xdim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int32]
        )
        
        return split_merged_relabeled_blob_id_field
    
    
    def split_and_merge_blobs(self, blob_id_field_unique, overlap_blobs_list, blob_props):
        '''Implements Blob Splitting & Merging.
           cf. Reference Paper
        
        Parameters:
        -----------
        blob_id_field_unique : xarray.DataArray
            Field of unique blob IDs. IDs must not be repeated across time.
        overlap_blobs_list : (N x 2) np.ndarray
            Array of overlapping blob pairs across time. The blob in the first column precedes the second column in time.
        blob_props : xarray.Dataset
            Properties of each blob, including 'area' and 'centroid'.
            
        Returns
        -------
        split_merged_blob_id_field_unique : xarray.DataArray
            Field of unique blob IDs with any splitting or merging logic applied.
        split_merged_blobs : (? x 2) np.ndarray
            Array of overlapping blob pairs, with any splitting or merging logic applied. May have removed or added pairs.
        '''
        
        
        # N.B. Need to modify the blob_id_field_unique as well... 
        #      Likely, need to add new labels, and change parts of certain labels in blob_id_field_unique, which then needs to be reflected in overlap_blobs_list
        
        
        
        
        return split_merged_blob_id_field_unique, split_merged_blobs
        
    
    
    def track_blObs(self, data_bin):
        '''Identifies & Labels Blobs across time, accounting for splitting & merging logic.
        
        Returns
        -------
        blob_id_field : xarray.DataArray
            Field of globally unique integer labels of each element in connected regions. ID = 0 indicates no object.
        '''
        
        ## Cluster & Label Binary Data at each Time Step
        blob_id_field, _ = self.identify_blobs(data_bin, time_connectivity=False)
        
        ## Generate Unique Blob IDs:  Add the cumulative time-sum to each blob ID to make them also Unique across Time
        cumulative_ids = (blob_id_field.max(dim={self.ydim, self.xdim}) + 1).cumsum(self.timedim).compute()
        min_int64 = np.iinfo(np.int64).min
        blob_id_field_adjust0 = blob_id_field.where(blob_id_field > 0, drop=False, other=min_int64)  # Need to protect the (unlabelled) 0s 
        blob_id_field_unique = (blob_id_field_adjust0 + cumulative_ids).persist()  # Persist because we will use this a lot...
                
        # Calculate Properties of each Blob
        blob_props = self.calculate_blob_properties(blob_id_field_unique, properties=['area', 'centroid'])
        
        # Compile List of Overlapping Blob ID Pairs Across Time
        overlap_blobs_list = self.find_overlapping_blobs(blob_id_field_unique)  # List of overlapping blob pairs
        
        # Apply Splitting & Merging Logic to `overlap_blobs`
        split_merged_blob_id_field_unique, split_merged_blobs_list = self.split_and_merge_blobs(blob_id_field_unique, overlap_blobs_list, blob_props)
                
        # Cluster Blobs List to Determine Globally Unique IDs & Update Blob ID Field
        split_merged_blob_id_field = self.cluster_rename_blobs(split_merged_blob_id_field_unique, split_merged_blobs_list)
        
        ## Count Number of Blobs (This may have increased due to splitting)
        N_blobs = split_merged_blob_id_field.max().compute().data
    
        return blob_id_field, N_blobs