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
    '''
    Spotter Identifies and Tracks Arbitrary Binary Blobs.
    '''
        
    def __init__(self, data_bin, mask, R_fill, area_filter_quartile, allow_merging=True, overlap_threshold=0.5, timedim='time', xdim='lat', ydim='lon'):
        
        self.data_bin           = data_bin
        self.mask               = mask
        self.R_fill             = int(R_fill)
        self.area_filter_quartile   = area_filter_quartile
        self.allow_merging      = allow_merging
        self.overlap_threshold  = overlap_threshold
        self.timedim    = timedim
        self.xdim       = xdim
        self.ydim       = ydim   
        
        if ((timedim, ydim, xdim) != data_bin.dims):
            try:
                data_bin = data_bin.transpose(timedim, ydim, xdim) 
            except:
                raise ValueError(f'Spot_the_blOb currently only supports 3D DataArrays. The dimensions should only contain ({timedim}, {xdim}, and {ydim}). Found {list(data_bin.dims)}')
        
        if (data_bin.data.dtype != bool):
            raise ValueError('The input DataArray is not binary. Please convert to a binary array, and try again.  :)')
        
        if not is_dask_collection(data_bin.data):
            raise ValueError('The input DataArray is not backed by a Dask array. Please chunk (in time), and try again.  :)')
        
        if (mask.data.dtype != bool):
            raise ValueError('The mask not binary. Please convert to a binary array, and try again.  :)')
        
        if (mask == False).all():
            raise ValueError('Found only False in `mask` input. The mask should indicate valid regions with True values.')
        
        if (area_filter_quartile < 0) or (area_filter_quartile > 1):
            raise ValueError('The discard_fraction should be between 0 and 1.')
        
            
    def run(self):
        '''
        Cluster, ID, filter, and track objects in a binary field with optional merging & splitting. 
        
        Parameters
        ----------
        data_bin : xarray.DataArray
            The _binary_ data to group & label. Must represent an underlying dask array.

        mask : xarray.DataArray
            The _binary_ mask of points to keep. False indicates points to ignore. 

        R_fill : int
            The size of the structuring element used in morphological opening & closing, relating to the largest hole that can be filled. In units of pixels.

        area_filter_quartile : float
            The fraction of the smallest objects to discard, i.e. the quantile defining the smallest area object retained. Value should be between 0 and 1.
        
        allow_merging : bool, optional
            Whether to allow splitting & merging of blobs across time. False defaults to classical `ndmeasure.label` with straight time connectivity.
        
        overlap_threshold : float, optional
            The minimum fraction of overlap between blobs across time to consider them the same object. Value should be between 0 and 1.
        
        Returns
        -------
        blob_id_field : xarray.DataArray
            Field of globally unique integer IDs of each element in connected regions. ID = 0 indicates no object.
        '''

        # Fill Small Holes & Gaps between Objects
        data_bin_filled = self.fill_holes()

        # Remove Small Objects
        data_bin_filtered, area_threshold, blob_areas, N_blobs_unfiltered = self.filter_small_blobs(data_bin_filled)

        if not self.allow_merging:
            # Track Blobs without any special merging or splitting
            blObs_ds, N_blobs_final = self.identify_blobs(data_bin_filtered, time_connectivity=True)
        else:
            # Track Blobs _with_ Merging & Splitting
            blObs_ds, N_blobs_final = self.track_blObs(data_bin_filtered)
        
        
        ## Save Some BlObby Stats

        total_id_area = int(blob_areas.sum().item())  # Percent of total object area retained after size filtering
        
        rejected_area = blob_areas.where(blob_areas <= area_threshold, drop=True).sum().item()
        rejected_area_fraction = rejected_area / total_id_area

        accepted_area = blob_areas.where(blob_areas > area_threshold, drop=True).sum().item()
        accepted_area_fraction = accepted_area / total_id_area

        blObs_ds.attrs['N_blobs_unfiltered'] = int(N_blobs_unfiltered)
        blObs_ds.attrs['N_blobs_final'] = int(N_blobs_final)
        blObs_ds.attrs['R_fill'] = self.R_fill
        blObs_ds.attrs['area_filter_quartile'] = self.area_filter_quartile
        blObs_ds.attrs['area_threshold'] = area_threshold
        blObs_ds.attrs['rejected_area_fraction'] = rejected_area_fraction
        blObs_ds.attrs['accepted_area_fraction'] = accepted_area_fraction
        

        ## Print Some BlObby Stats
        
        print(f'Total Object Area: {total_id_area}')
        print(f'Number of Initial Blobs: {N_blobs_unfiltered}')
        print(f'Area Cutoff Threshold: {area_threshold}')
        print(f'Rejected Area Fraction: {rejected_area_fraction}')
        print(f'Total Blobs Tracked: {N_blobs_final}')
        
        print(f'Total Merging Events: {len(blObs_ds.attrs["merge_ledger"])}')
        
        
        return blObs_ds
    
    

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
        y, x = np.ogrid[-self.R_fill:self.R_fill+1, -self.R_fill:self.R_fill+1]
        r = x**2 + y**2
        se_kernel = r < (self.R_fill**2)+1
        
        # Pad the binary data
        diameter = 2 * self.R_fill
        binary_data_padded = self.data_bin.pad({self.ydim: diameter, self.xdim: diameter, }, mode='wrap')
        
        # Apply binary closing and opening
        binary_data_closed = binary_closing(binary_data_padded.data, structure=se_kernel[np.newaxis, :, :])  # N.B.: Need to extract dask.array.Array from xarray.DataArray
        binary_data_opened = binary_opening(binary_data_closed, structure=se_kernel[np.newaxis, :, :])
        
        # Convert back to xarray.DataArray
        binary_data_opened = xr.DataArray(binary_data_opened, coords=binary_data_padded.coords, dims=binary_data_padded.dims)
        data_bin_filled    = binary_data_opened.isel({self.ydim: slice(diameter, -diameter), self.xdim: slice(diameter, -diameter)})
        
        # Mask out edge features arising from Morphological Operations
        data_bin_filled_mask = data_bin_filled.where(self.mask, drop=False, other=False)
        
        return data_bin_filled_mask


    def identify_blobs(self, data_bin, time_connectivity):
        '''IDs connected regions in the binary data.
        
        Returns
        -------
        blob_id_field : xarray.DataArray
            Field of integer IDs of each element in connected regions. ID = 0 indicates no object.
        '''
        
        neighbours = np.zeros((3,3,3))
        neighbours[1,:,:] = 1           # Connectivity Kernel: All 8 neighbours, but ignore time
        
        if time_connectivity:
            # ID blobs in 3D (i.e. space & time) -- N.B. IDs are unique across time
            neighbours[:,1,1] = 1 #                         including +-1 in time
        # else, ID blobs only in 2D (i.e. space) -- N.B. IDs are _not_ unique across time (i.e. each time starts at 0 again)    
        
        # Cluster & Label Binary Data
        blob_id_field, N_blobs = label(data_bin,           # Apply dask-powered ndimage & persist in memory
                                        structure=neighbours, 
                                        wrap_axes=(2,))       # Wrap in x-direction
        blob_id_field, N_blobs = persist(blob_id_field, N_blobs)
        
        N_blobs = N_blobs.compute()
        # DataArray (same shape as data_bin) but with integer IDs for each connected object: 
        blob_id_field = xr.DataArray(blob_id_field, coords=data_bin.coords, dims=data_bin.dims, attrs=data_bin.attrs).rename('ID_field')
        
        return blob_id_field, N_blobs
    
    
    def calculate_centroid(self, binary_mask, original_centroid=None):
        '''Re-calculates the centroid of a binary data blob if it touches the edges in the x-dimension.
        
        Parameters:
        -----------
        binary_mask : numpy.ndarray
            2D binary array where True indicates the blob
        original_centroid : tuple
            (y_centroid, x_centroid) from regionprops_table
            If None, the centroid is entirely calculated from the binary_mask.
            
        Returns:
        --------
        tuple
            (y_centroid, x_centroid)
        '''
        
        # Check if blob touches either edge of x dimension
        touches_left_BC = np.any(binary_mask[:, 0])
        touches_right_BC = np.any(binary_mask[:, -1])
        
        
        if original_centroid is None: # Then calculate y-centroid from scratch
            # Calculate y centroid
            y_indices = np.nonzero(binary_mask)[0]
            y_centroid = np.mean(y_indices)
        else: 
            y_centroid = original_centroid[1]
        
        
        # If blob touches both edges, then recalculate x-centroid
        if touches_left_BC and touches_right_BC:
            # Adjust x coordinates that are near right edge
            x_indices = np.nonzero(binary_mask)[1]
            x_indices_adj = x_indices.copy()
            right_side = x_indices > binary_mask.shape[1] // 2
            x_indices_adj[right_side] -= binary_mask.shape[1]
            
            x_centroid = np.mean(x_indices_adj)
            if x_centroid < 0:  # Ensure centroid is positive
                x_centroid += binary_mask.shape[1]
                
        elif original_centroid is None: # Then calculate x-centroid from scratch, as normal
            x_indices = np.nonzero(binary_mask)[1]
            x_centroid = np.mean(x_indices)
            
        else: 
            x_centroid = original_centroid[0]
        
        
        # N.B.: Returns original centroid if not touching both edges
        return (x_centroid, y_centroid)
        
    
    
    def calculate_blob_properties(self, blob_id_field, properties=None):
        '''
        Calculates properties of the blobs from the blob_id_field.
        
        Parameters:
        -----------
        blob_id_field : xarray.DataArray
            Field containing blob IDs
        properties : list, optional
            List of properties to calculate. If None, defaults to ['ID', 'area'].
            See skimage.measure.regionprops for available properties.
            
        Returns:
        --------
        xarray.Dataset
            Dataset containing all calculated properties with 'ID' dimension
        '''
        
        # Default Properties
        if properties is None:
            properties = ['label', 'area']
        
        # 'label' is needed for Identification
        if 'label' not in properties:
            properties = ['label'] + properties  # 'label' is actually 'ID' within regionprops
        
        check_centroids = 'centroid' in properties
        
        # Define wrapper function to run in parallel
        def blob_properties_chunk(ids):
            
            # Calculate Standard Properties
            props_slice = regionprops_table(ids, properties=properties)
            
            # Check Centroids if blob touches either edge (Need to account for x-dimension edge wrapping)
            if check_centroids and len(props_slice['label']) > 0:
                # Get original centroids
                centroids = list(zip(props_slice['centroid-0'], props_slice['centroid-1']))  # (x, y)
                centroids_wrapped = []
                
                # Process each blob
                for ID_idx, ID in enumerate(props_slice['label']):
                    binary_mask = ids == ID
                    centroids_wrapped.append(
                        self.calculate_centroid(binary_mask, centroids[ID_idx])
                    )
                
                # Update centroid values
                props_slice['centroid-0'] = [c[0] for c in centroids_wrapped]
                props_slice['centroid-1'] = [c[1] for c in centroids_wrapped]
            
            return props_slice
        
        if blob_id_field[self.timedim].size == 1:
            blob_props = blob_properties_chunk(blob_id_field.values.T)  # Makes x the first dimension (apply_ufunc flips it for some reason, so we need to match it...)
            blob_props = xr.Dataset({key: (['ID'], value) for key, value in blob_props.items()})
        else:
            # Run in parallel
        
            blob_props = xr.apply_ufunc(blob_properties_chunk, blob_id_field,
                                        input_core_dims=[[self.ydim, self.xdim]],
                                        output_core_dims=[[]],
                                        output_dtypes=[object],
                                        vectorize=True,
                                        dask='parallelized')
            
            # Concatenate and Convert to an xarray Dataset
            blob_props = xr.concat([
                xr.Dataset({key: (['ID'], value) for key, value in item.items()}) 
                for item in blob_props.values
            ], dim='ID')

        if blob_props.ID.size == 0:
            raise ValueError(f'No objects were detected.')
        
        # Combine centroid-0 and centroid-1 into a single centroid variable
        if 'centroid' in properties:
            blob_props['centroid'] = xr.concat([blob_props['centroid-0'], blob_props['centroid-1']], dim='component')
            blob_props = blob_props.drop_vars(['centroid-0', 'centroid-1'])
        
        # Set ID as coordinate
        blob_props = blob_props.set_index(ID='label')
        
        return blob_props
    

    def filter_small_blobs(self, data_bin):
        '''Filters out smallest ojects in the binary data.'''
        
        # Cluster & Label Binary Data: Time-independent in 2D (i.e. no time connectivity!)
        blob_id_field, N_blobs = self.identify_blobs(data_bin, time_connectivity=False)
        
        # Compute Blob Areas
        blob_props = self.calculate_blob_properties(blob_id_field)
        blob_areas, blob_ids = blob_props.area, blob_props.ID
        
        # Remove Smallest Blobs
        area_threshold = np.percentile(blob_areas, self.area_filter_quartile*100.0)
        blob_ids_keep = xr.where(blob_areas >= area_threshold, blob_ids, -1)
        blob_ids_keep[0] = -1  # Don't keep ID=0
        data_bin_filtered = blob_id_field.isin(blob_ids_keep)

        return data_bin_filtered, area_threshold, blob_areas, N_blobs
    
    def check_overlap_slice(self, ids_t0, ids_next):
            '''Finds overlapping blobs in a single time slice by looking at +-1 in time.'''
            
            # Flatten the arrays in a memory-efficient way
            ids_t0_flat = ids_t0.ravel()
            ids_next_flat = ids_next.ravel()
            
            # Create array of indices for valid IDs
            id_mask_t0 = ids_t0_flat > 0
            id_indices_t0 = np.nonzero(id_mask_t0)[0]
            id_indices_next = id_indices_t0[ids_next_flat[id_mask_t0] > 0]

            # Create Pairs using advanced indexing. N.B. We keep the earlier ID in time first
            id_pairs = np.stack((ids_t0_flat[id_indices_next],   ids_next_flat[id_indices_next]), axis=1)
            
            # Find unique pairs with their counts within this chunk
            unique_pairs, counts_chunk = np.unique(id_pairs, axis=0, return_counts=True)
            
            # Return pairs with their chunk-level counts
            return np.column_stack((unique_pairs, counts_chunk))
    
    def find_overlapping_blobs(self, blob_id_field):
        '''Finds overlapping blobs across time.
        
        Returns
        -------
        overlap_blobs_list_unique : (N x 3) np.ndarray
            Array of Blob IDs that indicate which blobs are overlapping in time. 
            The blob in the first column precedes the second column in time. 
            The third column is the number of overlapping cells.
        '''
        
        ## Check just for overlap with next time slice.
        #  Keep a running list of all blob IDs that overlap
        
        blob_id_field_next = blob_id_field.roll({self.timedim: -1}, roll_coords=False)

        # ID Overlapping Blobs in Parallel
        overlap_blob_pairs_list = xr.apply_ufunc(
                            self.check_overlap_slice,
                            blob_id_field,
                            blob_id_field_next,
                            input_core_dims=[[self.ydim, self.xdim], [self.ydim, self.xdim]],
                            output_core_dims=[[]],
                            vectorize=True,
                            dask="parallelized",
                            output_dtypes=[object]
                        ).compute()
        
        
        # Concatenate all pairs (with their chunk-level counts) from different chunks
        all_pairs_with_counts = np.concatenate(overlap_blob_pairs_list.values).astype(np.int32)
        
        # Get unique pairs and their indices
        unique_pairs, inverse_indices = np.unique(all_pairs_with_counts[:, :2], axis=0, return_inverse=True)

        # Sum the counts from the third column using the inverse indices
        total_summed_counts = np.zeros(len(unique_pairs), dtype=np.int32)
        np.add.at(total_summed_counts, inverse_indices, all_pairs_with_counts[:, 2])

        # Stack the pairs with their summed counts
        overlap_blobs_list_unique = np.column_stack((unique_pairs, total_summed_counts))
        
        return overlap_blobs_list_unique
    
        
    def cluster_rename_blobs_and_props(self, blob_id_field_unique, blobs_props, overlap_blobs_list):
        '''Cluster the blob pairs to determine the final IDs, and relabel the blobs.
        
        Parameters
        ----------
        blob_id_field_unique : xarray.DataArray
            Field of unique blob IDs. IDs must not be repeated across time.
        blobs_props : xarray.Dataset
            Properties of each blob that also need to be relabeled.
        overlap_blobs_list : (N x 2) np.ndarray
            Array of Blob IDs that indicate which blobs are the same. The blob in the first column precedes the second column in time.
        
        Returns
        -------
        Merged DataSet including:
            split_merged_relabeled_blob_id_field : xarray.DataArray
                Field of renamed blob IDs which track & ID blobs across time. ID = 0 indicates no object.
            blobs_props_extended : xarray.Dataset
                Properties of each blob, with the updated IDs.
                Contains all original properties, as well as "global_ID" (the original ID), and which puts blobs & properties in the time-dimension
        '''
        
        
        ## Cluster the overlap_pairs into groups of IDs that are actually the same blob
        
        # Get unique IDs from the overlap pairs
        IDs = np.unique(overlap_blobs_list) # 1D sorted unique
        
        # Create a mapping from ID to indices
        ID_to_index = {ID: index for index, ID in enumerate(IDs)}
        
        # Convert overlap pairs to indices
        overlap_pairs_indices = np.array([(ID_to_index[pair[0]], ID_to_index[pair[1]]) for pair in overlap_blobs_list])
        
        # Create a sparse matrix representation of the graph
        n = len(IDs)
        row_indices, col_indices = overlap_pairs_indices.T
        data = np.ones(len(overlap_pairs_indices))
        graph = csr_matrix((data, (row_indices, col_indices)), shape=(n, n))
        
        # Solve the graph to determine connected components
        num_components, component_IDs = connected_components(csgraph=graph, directed=False, return_labels=True)
        
        # Group IDs by their component index
        ID_clusters = [[] for _ in range(num_components)]
        for ID, component_ID in zip(IDs, component_IDs):
            ID_clusters[component_ID].append(ID)
        
        
        
        ## ID_clusters now is a list of lists of equivalent blob IDs that have been tracked across time
        #  We now need to replace all IDs in blob_id_field_unique that match the equivalent_IDs with the list index:  This is the new/final ID field.
        
        # Create a dictionary to map IDs to the new cluster indices
        min_int32 = np.iinfo(np.int32).min
        max_old_ID = blob_id_field_unique.max().compute().data
        ID_to_cluster_index_array = np.full(max_old_ID + 1, min_int32, dtype=np.int32)

        # Fill the lookup array with cluster indices
        for index, cluster in enumerate(ID_clusters):
            for ID in cluster:
                ID_to_cluster_index_array[ID] = np.int32(index+1) # Because these are the connected IDs, there are many fewer!
                                                                  #  Add 1 so that ID = 0 is still invalid/no object
        
        # N.B.: **Need to pass da into apply_ufunc, otherwise it doesn't manage the memory correctly with large shared-mem numpy arrays**
        ID_to_cluster_index_da = xr.DataArray(ID_to_cluster_index_array, dims='ID', coords={'ID': np.arange(max_old_ID + 1)})
        
        def map_IDs_to_indices(block, ID_to_cluster_index_array):
            mask = block > 0
            new_block = np.zeros_like(block, dtype=np.int32)
            new_block[mask] = ID_to_cluster_index_array[block[mask]]
            return new_block
        
        split_merged_relabeled_blob_id_field = xr.apply_ufunc(
            map_IDs_to_indices,
            blob_id_field_unique, 
            ID_to_cluster_index_da,
            input_core_dims=[[self.ydim, self.xdim],['ID']],
            output_core_dims=[[self.ydim, self.xdim]],
            vectorize=True,
            dask="parallelized",
            output_dtypes=[np.int32]
        )
        
        
        
        ### Relabel the blobs_props to match the new IDs (and add time dimension!)
        
        max_new_ID = num_components + 1  # New IDs range from 0 to max_new_ID...
        new_ids = np.arange(1, max_new_ID+1)
        
        # New blobs_props DataSet Structure
        blobs_props_extended = xr.Dataset(coords={
            'ID': new_ids,
            self.timedim: blob_id_field_unique[self.timedim]
        })
        
        ## Create a mapping from new IDs to the original IDs _at the corresponding time_
        valid_new_ids = (split_merged_relabeled_blob_id_field > 0)      
        original_ids = blob_id_field_unique.where(valid_new_ids).stack(z=(self.ydim, self.xdim))
        new_ids_field = split_merged_relabeled_blob_id_field.where(valid_new_ids).stack(z=(self.ydim, self.xdim))
        
        id_mapping = xr.Dataset({
            'original_id': original_ids,
            'new_id': new_ids_field
        }).dropna(dim='z')
        
        global_id_mapping = id_mapping.groupby('new_id').original_id.first().rename({'new_id': 'ID'})
        blobs_props_extended['global_ID'] = global_id_mapping
        # N.B.: Now, e.g. global_id_mapping.sel(ID=10) --> Given the new ID (10), returns corresponding original_id at every time
        
        
        ## Transfer and transform all variables from original blobs_props:
        
        # Make a map from (old)ID ---> time & (new)ID
        old_id_to_new_id = xr.DataArray(
            data=id_mapping.original_id.values,
            coords={
                self.timedim: ('z', id_mapping[self.timedim].values),
                'ID': ('z', id_mapping.new_id.values)
            },
            dims=['z']
        )
        
        for var_name in blobs_props.data_vars:
            
            var_transformed = (blobs_props[var_name]
                            .sel(ID=old_id_to_new_id)               # Select original property values using original IDs
                            .assign_coords(ID=old_id_to_new_id.ID)  # Replace original IDs with new IDs
                            .groupby(self.timedim)                        # Group by time to reshape
                            .first()                                # Take first value if multiple exist
                            .reindex(ID=new_ids))                   # Ensure all new IDs are present

            blobs_props_extended[var_name] = var_transformed
        
        
        # Add start and end time indices for each ID
        valid_presence = (~blobs_props_extended['global_ID'].isnull())  # Where we have valid data
        
        blobs_props_extended['time_start'] = valid_presence.where(valid_presence)[self.timedim].min(dim=self.timedim)
        blobs_props_extended['time_end'] = valid_presence.where(valid_presence)[self.timedim].max(dim=self.timedim)
        
        # Combine blobs_props_extended with split_merged_relabeled_blob_id_field
        split_merged_relabeled_blobs_ds = xr.merge([split_merged_relabeled_blob_id_field.rename('ID_field'), 
                                                 blobs_props_extended])
                
        
        return split_merged_relabeled_blobs_ds
    
    
    
    def split_and_merge_blobs(self, blob_id_field_unique, blob_props, overlap_blobs_list):
        '''Implements Blob Splitting & Merging.
           cf. Reference Di Sun  & Bohai Zhang 2023
        
        Parameters:
        -----------
        blob_id_field_unique : xarray.DataArray
            Field of unique blob IDs. IDs must not be repeated across time.
        overlap_blobs_list : (N x 3) np.ndarray
            Array of overlapping blob pairs across time. 
            The blob in the first column precedes the second column in time.
            The third column is the number of overlapping cells.
        blob_props : xarray.Dataset
            Properties of each blob, including 'area' and 'centroid'.
            
        Returns
        -------
        split_merged_blob_id_field_unique : xarray.DataArray
            Field of unique blob IDs with any splitting or merging logic applied.
        merged_blob_props : xarray.Dataset
            Properties of each blob, now containing any new blob IDs & properties
        split_merged_blobs_list : (? x 2) np.ndarray
            Array of overlapping blob pairs, with any splitting or merging logic applied. May have removed or added pairs.
            3rd column of overlap area is unnecessary and is removed.
        merge_ledger : list
            List of tuples indicating which blobs have been merged.
        '''
        
        
        ##### Enforce all Blob Pairs overlap by at least 50% (in Area)
        
        # Vectorised computation of overlap fractions
        areas_0 = blob_props['area'].sel(ID=overlap_blobs_list[:, 0]).values
        areas_1 = blob_props['area'].sel(ID=overlap_blobs_list[:, 1]).values
        min_areas = np.minimum(areas_0, areas_1)
        overlap_fractions = overlap_blobs_list[:, 2].astype(float) / min_areas

        # Filter out the overlaps that are too small
        overlap_blobs_list = overlap_blobs_list[overlap_fractions >= self.overlap_threshold]
        
        
        
        ##### Consider Merging Blobs
        
        # Initialise merge tracking structures
        merge_ledger = []                      # List of IDs of the 2 Merging Parents
        next_new_id = blob_props.ID.max().item() + 1  # Start new IDs after highest existing ID
        
        # Find all the Children (t+1 / RHS) elements that appear multiple times --> Indicates there are 2+ Parent Blobs...
        unique_children, children_counts = np.unique(overlap_blobs_list[:, 1], return_counts=True)
        merging_blobs = unique_children[children_counts > 1]
        
        
        for child_id in merging_blobs:
            
            # Find all pairs involving this Child Blob
            child_mask = overlap_blobs_list[:, 1] == child_id
            child_where = np.where(overlap_blobs_list[:, 1] == child_id)[0]  # Needed for assignment
            merge_group = overlap_blobs_list[child_mask]
            
            # Get all Parents (LHS) Blobs that overlap with this Child Blob -- N.B. This is now generalised for N-parent merging !
            parent_ids = merge_group[:, 0]
            num_parents = len(parent_ids)
            
            # Make a new ID for the other Half of the Child Blob & Record in the Merge Ledger
            new_blob_id = np.arange(next_new_id, next_new_id + (num_parents - 1), dtype=np.int32)
            next_new_id += num_parents - 1
            merge_ledger.append(parent_ids)
            
            # Replace the 2nd+ Child in the Overlap Blobs List with the new Child ID
            overlap_blobs_list[child_where[1:], 1] = new_blob_id    #overlap_blobs_list[child_mask, 1][1:] = new_blob_id
            child_ids = np.concatenate((np.array([child_id]), new_blob_id))    #np.array([child_id, new_blob_id])
            
            ## Relabel the Original Child Blob ID Field to account for the New ID:
            # --> For every (Original) Child Cell in the ID Field, Measure the Distance to the Centroids of the Parents
            # --> Assign the ID for each Cell corresponding to the closest Parent
            
            parent_centroids = blob_props.sel(ID=parent_ids).centroid.isel(component=[1,0]).values.T  # (y, x), [:,0] are the y's
            child_time_idx = (blob_id_field_unique == child_id).any(dim=[self.ydim, self.xdim]).argmax().compute().item()
            child_mask_2d = blob_id_field_unique.isel({self.timedim: child_time_idx}) == child_id
            child_coords = np.stack(np.where(child_mask_2d), axis=1)
            distances = np.linalg.norm(child_coords[:, None] - parent_centroids, axis=2) # (n_child_cells, 2)
            
            # Assign the new ID to each cell based on the closest parent
            new_labels = child_ids[np.argmin(distances, axis=1)]
            
            # Update values in child_time_idx and assign the updated slice back to the original DataArray
            tslice_child = blob_id_field_unique.isel({self.timedim: child_time_idx})
            temp = np.zeros_like(tslice_child)
            temp[child_mask_2d] = new_labels
            blob_id_field_unique[{self.timedim: child_time_idx}] = tslice_child.where(~child_mask_2d, temp)
            
            
            ## Update the Properties of the N Children Blobs
            new_child_props = self.calculate_blob_properties(blob_id_field_unique.isel({self.timedim: child_time_idx}), properties=['area', 'centroid'])
            
            # Update the properties for the original child ID
            blob_props.loc[dict(ID=child_id)] = new_child_props.sel(ID=child_id)
            
            # Add the properties for the N-1 other new child ID
            blob_props = xr.concat([blob_props, new_child_props.sel(ID=new_blob_id)], dim='ID')
        

            ## Finally, we need to re-assess all of the Parent IDs (LHS) equal to the (original) child_id
            
            # Look at the overlap IDs between the original child_id and the next time-step, and also the new_blob_id and the next time-step
            new_overlaps = self.check_overlap_slice(blob_id_field_unique.isel({self.timedim: child_time_idx}).values, blob_id_field_unique.isel({self.timedim: child_time_idx+1}).values)
            new_child_overlaps_list = new_overlaps[(new_overlaps[:, 0] == child_id) | np.isin(new_overlaps[:, 0], new_blob_id)]
            
            # _Before_ replacing the overlap_blobs_list, we need to re-assess the overlap fractions of just the new_child_overlaps_list
            areas_0 = blob_props['area'].sel(ID=new_child_overlaps_list[:, 0]).values
            areas_1 = blob_props['area'].sel(ID=new_child_overlaps_list[:, 1]).values
            min_areas = np.minimum(areas_0, areas_1)
            overlap_fractions = new_child_overlaps_list[:, 2].astype(float) / min_areas
            new_child_overlaps_list = new_child_overlaps_list[overlap_fractions >= self.overlap_threshold]
            
            # Replace the lines in the overlap_blobs_list where (original) child_id is on the LHS, with these new pairs in new_child_overlaps_list
            child_mask_LHS = overlap_blobs_list[:, 0] == child_id
            overlap_blobs_list = np.concatenate([overlap_blobs_list[~child_mask_LHS], new_child_overlaps_list])
            
        
        return (blob_id_field_unique, 
                blob_props, 
                overlap_blobs_list[:, :2],
                merge_ledger)
        
    
    
    def track_blObs(self, data_bin):
        '''Identifies Blobs across time, accounting for splitting & merging logic.
        
        Returns
        -------
        blob_id_field : xarray.DataArray
            Field of globally unique integer IDs of each element in connected regions. ID = 0 indicates no object.
        '''
        
        # Cluster & ID Binary Data at each Time Step
        blob_id_field, _ = self.identify_blobs(data_bin, time_connectivity=False)
                
        # Calculate Properties of each Blob
        blob_props = self.calculate_blob_properties(blob_id_field, properties=['area', 'centroid'])
        
        # Compile List of Overlapping Blob ID Pairs Across Time
        overlap_blobs_list = self.find_overlapping_blobs(blob_id_field)  # List of overlapping blob pairs
        
        # Apply Splitting & Merging Logic to `overlap_blobs`
        split_merged_blob_id_field_unique, merged_blobs_props, split_merged_blobs_list, merged_blobs_ledger = self.split_and_merge_blobs(blob_id_field, blob_props, overlap_blobs_list)
        
        # Cluster Blobs List to Determine Globally Unique IDs & Update Blob ID Field
        split_merged_blobs_ds = self.cluster_rename_blobs_and_props(split_merged_blob_id_field_unique, merged_blobs_props, split_merged_blobs_list)
        
        # Add Merge Ledger to split_merged_blobs_ds
        split_merged_blobs_ds.attrs['merge_ledger'] = merged_blobs_ledger
        
        # Count Number of Blobs (This may have increased due to splitting)
        N_blobs = split_merged_blobs_ds.ID_field.max().compute().data
    
        return split_merged_blobs_ds, N_blobs
    

    