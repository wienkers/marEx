import xarray as xr
import numpy as np
from dask_image.ndmeasure import label
from skimage.measure import regionprops_table
from dask_image.ndmorph import binary_closing as binary_closing_dask
from dask_image.ndmorph import binary_opening as binary_opening_dask
from scipy.ndimage import binary_closing, binary_opening
from scipy.sparse import csr_matrix
from scipy.sparse.csgraph import connected_components
from dask import persist
from dask.base import is_dask_collection
from numba import jit, prange
import warnings

class Spotter:
    '''
    Spotter Identifies and Tracks Arbitrary Binary Blobs.
    '''
        
    def __init__(self, data_bin, mask, R_fill, area_filter_quartile, allow_merging=True, nn_partitioning=False, overlap_threshold=0.5, timedim='time', xdim='lon', ydim='lat'):
        
        self.data_bin           = data_bin
        self.mask               = mask
        self.R_fill             = int(R_fill)
        self.area_filter_quartile   = area_filter_quartile
        self.allow_merging      = allow_merging
        self.nn_partitioning = nn_partitioning
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
        
            
    def run(self, return_merges=False):
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
            Whether to allow splitting & merging of blobs across time. False defaults to classical `ndmeasure.label` with straight time connectivity, i.e. Scannell et al. 
        
        nn_partitioning : bool, optional
            If True, then implement a better merged child partitioning calculation based on closest parent cell.
            If False, then the centroid is used to determine partitioning between new child labels, e.g. Di Sun & Bohai Zhang 2023
        
        overlap_threshold : float, optional
            The minimum fraction of overlap between blobs across time to consider them the same object. Value should be between 0 and 1.
        
        Returns
        -------
        blob_id_field : xarray.DataArray
            Field of globally unique integer IDs of each element in connected regions. ID = 0 indicates no object.
        '''

        # Fill Small Holes & Gaps between Objects
        data_bin_filled = self.fill_holes()
        print('Finished filling holes.')

        # Remove Small Objects
        data_bin_filtered, area_threshold, blob_areas, N_blobs_unfiltered = self.filter_small_blobs(data_bin_filled)
        print('Finished filtering small blobs.')
        
        if not self.allow_merging:
            # Track Blobs without any special merging or splitting
            blObs_ds, N_blobs_final = self.identify_blobs(data_bin_filtered, time_connectivity=True)
        else:
            # Track Blobs _with_ Merging & Splitting
            blObs_ds, merges_ds, N_blobs_final = self.track_blObs(data_bin_filtered)
        print('Finished tracking blobs.')
        
        
        ## Save Some BlObby Stats

        total_id_area = int(blob_areas.sum().item())  # Percent of total object area retained after size filtering
        
        rejected_area = blob_areas.where(blob_areas <= area_threshold, drop=True).sum().item()
        rejected_area_fraction = rejected_area / total_id_area

        accepted_area = blob_areas.where(blob_areas > area_threshold, drop=True).sum().item()
        accepted_area_fraction = accepted_area / total_id_area

        blObs_ds.attrs['allow_merging'] = int(self.allow_merging)
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
        
        if self.allow_merging:
            
            
            blObs_ds.attrs['overlap_threshold'] = self.overlap_threshold
            blObs_ds.attrs['nn_partitioning'] = int(self.nn_partitioning)
            
            # Add merge-specific summary attributes 
            blObs_ds.attrs['total_merges'] = len(merges_ds.merge_ID)
            blObs_ds.attrs['multi_parent_merges'] = (merges_ds.n_parents > 2).sum().item()
            
            print(f"Total Merging Events: {blObs_ds.attrs['total_merges']}")
            print(f"Multi-Parent Merging Events: {blObs_ds.attrs['multi_parent_merges']}")
        
        
        if self.allow_merging and return_merges:
            return blObs_ds, merges_ds
        else:
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
        
        use_scipy_morph = True  # dask_image.ndmorph currently has a bug in the binary_closing function
        
        # Generate Structuring Element
        y, x = np.ogrid[-self.R_fill:self.R_fill+1, -self.R_fill:self.R_fill+1]
        r = x**2 + y**2
        diameter = 2 * self.R_fill
        se_kernel = r < (self.R_fill**2)+1
        
        
        if use_scipy_morph:
            
            def binary_open_close(bitmap_binary):
                bitmap_binary_padded = np.pad(bitmap_binary,
                                                ((diameter, diameter), (diameter, diameter)),
                                                mode='wrap')
                s1 = binary_closing(bitmap_binary_padded, se_kernel, iterations=1)
                s2 = binary_opening(s1, se_kernel, iterations=1)
                unpadded= s2[diameter:-diameter, diameter:-diameter]
                return unpadded

            data_bin_filled = xr.apply_ufunc(binary_open_close, self.data_bin,
                                    input_core_dims=[[self.ydim, self.xdim]],
                                    output_core_dims=[[self.ydim, self.xdim]],
                                    output_dtypes=[self.data_bin.dtype],
                                    vectorize=True,
                                    dask='parallelized')
        
        else:  # _CAUTION_:  Optimised dask_ndimage library gives some incorrectly holy/segmented binary-closed images...
        
            binary_data_padded = self.data_bin.pad({self.ydim: diameter, self.xdim: diameter, }, mode='wrap')
            binary_data_closed = binary_closing_dask(binary_data_padded.data, structure=se_kernel[np.newaxis, :, :])  # N.B.: Need to extract dask.array.Array from xarray.DataArray
            binary_data_opened = binary_opening_dask(binary_data_closed, structure=se_kernel[np.newaxis, :, :])
            
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
            neighbours[:,:,:] = 1 #                         including +-1 in time, _and also diagonal in time_ -- i.e. edges can touch
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
            2D binary array where True indicates the blob. Dimensions are _exceptionally_ ordered (y,x) here...
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
            y_centroid = original_centroid[0]
        
        
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
            x_centroid = original_centroid[1]
        
        
        # N.B.: Returns original centroid if not touching both edges
        return (y_centroid, x_centroid)
        
    
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
            # N.B. Assumes the dimensions are ordered (y, x)
            
            # Calculate Standard Properties
            props_slice = regionprops_table(ids, properties=properties)
            
            # Check Centroids if blob touches either edge (Need to account for x-dimension edge wrapping)
            if check_centroids and len(props_slice['label']) > 0:
                # Get original centroids
                centroids = list(zip(props_slice['centroid-0'], props_slice['centroid-1']))  # (y, x)
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
            blob_props = blob_properties_chunk(blob_id_field.values)
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
        
        # Create masks for valid IDs
        mask_t0 = ids_t0 > 0
        mask_next = ids_next > 0
        
        # Only process cells where both times have valid IDs
        combined_mask = mask_t0 & mask_next
        
        if not np.any(combined_mask):
            return np.empty((0, 3), dtype=np.int32)
        
        # Extract only the overlapping points
        ids_t0_valid = ids_t0[combined_mask].astype(np.int64)
        ids_next_valid = ids_next[combined_mask].astype(np.int64)
        
        # Create a unique identifier for each pair
        # This is faster than using np.unique with axis=1
        max_id = max(ids_t0.max(), ids_next.max()).astype(np.int64) # N.B.: If more IDs than ~srqt(largest int64), then this will give problems.....
        pair_ids = ids_t0_valid * (max_id + 1) + ids_next_valid
        
        # Get unique pairs and their counts
        unique_pairs, counts = np.unique(pair_ids, return_counts=True)
        
        # Convert back to original ID pairs
        id_t0 = (unique_pairs // (max_id + 1)).astype(np.int32)
        id_next = (unique_pairs % (max_id + 1)).astype(np.int32)
        
        # Stack results
        result = np.column_stack((id_t0, id_next, counts.astype(np.int32)))
        
        return result.astype(np.int32)
    
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
        
        blob_id_field_next = blob_id_field.shift({self.timedim: -1}, fill_value=0)

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
    
        
    def cluster_rename_blobs_and_props(self, blob_id_field_unique, blobs_props, overlap_blobs_list, merge_events):
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
        new_ids = np.arange(1, max_new_ID+1, dtype=np.int32)
        
        # New blobs_props DataSet Structure
        blobs_props_extended = xr.Dataset(coords={
            'ID': new_ids,
            self.timedim: blob_id_field_unique[self.timedim]
        })
        
        ## Create a mapping from new IDs to the original IDs _at the corresponding time_
        valid_new_ids = (split_merged_relabeled_blob_id_field > 0)      
        original_ids_field = blob_id_field_unique.where(valid_new_ids).stack(z=(self.ydim, self.xdim), create_index=False)
        new_ids_field = split_merged_relabeled_blob_id_field.where(valid_new_ids).stack(z=(self.ydim, self.xdim), create_index=False)
        
        new_ids_da = xr.DataArray(new_ids, dims='new_id').chunk({'new_id': 400})
        first_match_idx = (new_ids_field.chunk({'z': 50000}) == new_ids_da).argmax(dim='z').compute()
        
        result = xr.where(new_ids_field.isel(z=first_match_idx) == new_ids, 
                          original_ids_field.isel(z=first_match_idx), 0)
        
        global_id_mapping = (result
            .assign_coords(new_id=new_ids)
            .rename({'new_id': 'ID'})
            .astype(np.int32)
            .compute())

        blobs_props_extended['global_ID'] = global_id_mapping
        # N.B.: Now, e.g. global_id_mapping.sel(ID=10) --> Given the new ID (10), returns corresponding original_id at every time
        
        
        ## Transfer and transform all variables from original blobs_props:
                
        # Add a value of ID = 0 to this coordinate ID
        dummy = blobs_props.isel(ID=0) * np.nan
        blobs_props = xr.concat([dummy.assign_coords(ID=0), blobs_props], dim='ID')
        
        
        for var_name in blobs_props.data_vars:
            
            temp = (blobs_props[var_name]
                              .sel(ID=global_id_mapping.rename({'ID':'new_id'}))
                              .drop_vars('ID').rename({'new_id':'ID'}))
            
            if var_name == 'ID':
                temp = temp.astype(np.int32)
            else:
                temp = temp.astype(np.float32)
                
            blobs_props_extended[var_name] = temp
        
        
        ## Map the merge_events using the old IDs to be from dimensions (merge_ID, parent_idx) 
        #     --> new merge_ledger with dimensions (time, ID, sibling_ID)
        # i.e. for each merge_ID --> merge_parent_IDs   gives the old IDs  --> map to new ID using ID_to_cluster_index_da
        #                   --> merge_time
        
        old_parent_IDs = xr.where(merge_events.parent_IDs>0, merge_events.parent_IDs, 0)
        new_IDs_parents = ID_to_cluster_index_da.sel(ID=old_parent_IDs)

        # Replace the coordinate merge_ID in new_IDs_parents with merge_time.  merge_events.merge_time gives merge_time for each merge_ID
        new_IDs_parents_t = new_IDs_parents.assign_coords({'merge_time': merge_events.merge_time}).drop_vars('ID').swap_dims({'merge_ID': 'merge_time'})  # this now has coordinate merge_time and ID

        # Map new_IDs_parents_t into a new data array with dimensions time, ID, and sibling_ID
        merge_ledger = xr.full_like(global_id_mapping, fill_value=-1).expand_dims({'sibling_ID': new_IDs_parents_t.parent_idx.shape[0]}).copy() # dimesions are time, ID, sibling_ID
        
        for time_val in new_IDs_parents_t.merge_time.values:
            IDs = new_IDs_parents_t.sel({'merge_time': time_val})
            if IDs.ndim == 1:
                IDs = IDs.values
                for ID in IDs:
                    if ID > 0:
                        merge_ledger.loc[{self.timedim: time_val, 'ID': ID}] = IDs
            else:  # There were multiple mergers at this time...
                for merge_num, _ in enumerate(IDs.merge_time):
                    IDs_sub = IDs.isel(merge_time=merge_num).values
                    for ID in IDs_sub:
                        if ID > 0:
                            merge_ledger.loc[{self.timedim: time_val, 'ID': ID}] = IDs_sub
        
        merge_ledger = merge_ledger.rename('merge_ledger').transpose(self.timedim, 'ID', 'sibling_ID').chunk({self.timedim: split_merged_relabeled_blob_id_field.data.chunksize[0]})
        
        
        ## Finish up:
        # Add start and end time indices for each ID
        valid_presence = blobs_props_extended['global_ID'] > 0  # Where we have valid data
        
        blobs_props_extended['presence'] = valid_presence
        blobs_props_extended['time_start'] = valid_presence.time[valid_presence.argmax(dim=self.timedim)]
        blobs_props_extended['time_end'] = valid_presence.time[(valid_presence.sizes[self.timedim] - 1) - (valid_presence[::-1]).argmax(dim=self.timedim)]
                
        # Combine blobs_props_extended with split_merged_relabeled_blob_id_field
        split_merged_relabeled_blobs_ds = xr.merge([split_merged_relabeled_blob_id_field.rename('ID_field'), 
                                                    blobs_props_extended,
                                                    merge_ledger])
        
        
        # Remove the last ID -- it is all 0s        
        return split_merged_relabeled_blobs_ds.isel(ID=slice(0, -1))
    
    
    def compute_id_time_dict(self, da, child_blobs, max_blobs, all_blobs=True):
        '''Generate lookup table mapping blob IDs to their time index.
        
        Parameters
        ----------
        da : xarray.DataArray
            Field of unique blob IDs. IDs must not be repeated across time.
        child_blobs: np.ndarray
            Array of blob IDs to be the dictionary keys.
        max_blobs: int
            Maximum number of blobs in the entire data array
        '''
            
        # First reduce x & y dimensions by determining unique IDs for each time slice
        est_blobs_per_time_max = int(max_blobs / da[self.timedim].shape[0] * 100)

        def unique_pad(x):
            uniq = np.unique(x)
            result = np.zeros(est_blobs_per_time_max, dtype=x.dtype) # Pad output to maximum size
            result[:len(uniq)] = uniq
            return result

        unique_ids_by_time = xr.apply_ufunc(
                unique_pad,
                da,
                input_core_dims=[[self.ydim, self.xdim]],
                output_core_dims=[['unique_values']],
                dask='parallelized',
                vectorize=True,
                dask_gufunc_kwargs={'output_sizes': {'unique_values': est_blobs_per_time_max}}
            )
        
        if not all_blobs:  # Just index the blobs in "child_blobs"
            search_ids = xr.DataArray(
                child_blobs,
                dims=['child_id'],
                coords={'child_id': child_blobs}
            )
        else:
            search_ids = xr.DataArray(
                np.arange(max_blobs, dtype=np.int32),
                dims=['child_id'],
                coords={'child_id': np.arange(max_blobs)}
            ).chunk({'child_id': 10000}) # ~ max_blobs // 20
            

        # Reduce boolean array in spatial dimensions for all IDs at once
        time_indices = ((unique_ids_by_time == search_ids)
                .any(dim=['unique_values'])   # Reduce along the unique-dim 
                .argmax(dim=self.timedim).compute())  # Get the time-index where true
        
        # Convert to dictionary for fast lookup
        time_index_map = {
            int(id_val): int(idx.values) 
            for id_val, idx in zip(time_indices.child_id, time_indices)
        }
        
        return time_index_map
    
    
    def split_and_merge_blobs(self, blob_id_field_unique, blob_props, overlap_blobs_list):
        '''Implements Blob Splitting & Merging.
        
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
        
        ###################################################################
        ##### Enforce all Blob Pairs overlap by at least 50% (in Area) ####
        ###################################################################
        
        ## Vectorised computation of overlap fractions
        areas_0 = blob_props['area'].sel(ID=overlap_blobs_list[:, 0]).values
        areas_1 = blob_props['area'].sel(ID=overlap_blobs_list[:, 1]).values
        min_areas = np.minimum(areas_0, areas_1)
        overlap_fractions = overlap_blobs_list[:, 2].astype(float) / min_areas

        ## Filter out the overlaps that are too small
        overlap_blobs_list = overlap_blobs_list[overlap_fractions >= self.overlap_threshold]
        
        
        
        #################################
        ##### Consider Merging Blobs ####
        #################################
        
        ## Initialize merge tracking lists to build DataArray later
        merge_times = []      # When the merge occurred
        merge_child_ids = []  # Resulting child ID
        merge_parent_ids = [] # List of parent IDs that merged
        merge_areas = []      # Areas of overlap
        next_new_id = blob_props.ID.max().item() + 1  # Start new IDs after highest existing ID
        
        # Find all the Children (t+1 / RHS) elements that appear multiple times --> Indicates there are 2+ Parent Blobs...
        unique_children, children_counts = np.unique(overlap_blobs_list[:, 1], return_counts=True)
        merging_blobs = unique_children[children_counts > 1]
        
        # Pre-compute the child_time_idx & 2d_mask_id for each child_blob
        time_index_map = self.compute_id_time_dict(blob_id_field_unique, merging_blobs, next_new_id)
        Nx = blob_id_field_unique[self.xdim].size
        
        # Group blobs by time-chunk
        # -- Pre-condition: Blob IDs should be monotonically increasing in time...
        chunk_boundaries = np.cumsum([0] + list(blob_id_field_unique.chunks[0] ))
        blobs_by_chunk = {}
        # Ensure that blobs_by_chunk has entry for every key
        for chunk_idx in range(len(blob_id_field_unique.chunks[0])):
            blobs_by_chunk.setdefault(chunk_idx, [])
        
        for blob_id in merging_blobs:
            # Find which chunk this time index belongs to
            chunk_idx = np.searchsorted(chunk_boundaries, time_index_map[blob_id], side='right') - 1
            blobs_by_chunk.setdefault(chunk_idx, []).append(blob_id)
        
        
        future_chunk_merges = []
        for chunk_idx, chunk_blobs in blobs_by_chunk.items(): # Loop over each time-chunk
            # We do this to avoid repetetively re-computing and injecting tiny changes into the full dask-backed DataArray blob_id_field_unique
            
            ## Extract and Load an entire chunk into memory
            
            chunk_start = sum(blob_id_field_unique.chunks[0][:chunk_idx])
            chunk_end = chunk_start + blob_id_field_unique.chunks[0][chunk_idx] + 1  #  We also want access to the blob_id_time_p1...  But need to remember to remove the last time later
            
            chunk_data = blob_id_field_unique.isel({self.timedim: slice(chunk_start, chunk_end)}).compute()
            
            # Create a working queue of blobs to process
            blobs_to_process = chunk_blobs.copy()
            # Combine only the future_chunk_merges that don't already appear in blobs_to_process
            blobs_to_process = blobs_to_process + [blob_id for blob_id in future_chunk_merges if blob_id not in blobs_to_process]  # First, assess the new blobs from the end of the previous chunk...
            future_chunk_merges = []
            
            #for child_id in chunk_blobs: # Process each blob in this chunk
            while blobs_to_process:  # Process until queue is empty
                child_id = blobs_to_process.pop(0)  # Get next blob to process
                
                child_time_idx = time_index_map[child_id]
                relative_time_idx = child_time_idx - chunk_start
                
                blob_id_time = chunk_data.isel({self.timedim: relative_time_idx})
                try:
                    blob_id_time_p1 = chunk_data.isel({self.timedim: relative_time_idx+1})
                except: # If this is the last chunk...
                    blob_id_time_p1 = xr.full_like(blob_id_time, 0)
                if relative_time_idx-1 >= 0:
                    blob_id_time_m1 = chunk_data.isel({self.timedim: relative_time_idx-1})
                elif chunk_idx > 0:
                    blob_id_time_m1 = blob_id_field_unique.isel({self.timedim: chunk_start-1})
                else:
                    blob_id_time_m1 = xr.full_like(blob_id_time, 0)
                
                child_mask_2d  = (blob_id_time == child_id).values
                
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
                
                # Replace the 2nd+ Child in the Overlap Blobs List with the new Child ID
                overlap_blobs_list[child_where[1:], 1] = new_blob_id    #overlap_blobs_list[child_mask, 1][1:] = new_blob_id
                child_ids = np.concatenate((np.array([child_id]), new_blob_id))    #np.array([child_id, new_blob_id])
                
                # Record merge event data
                merge_times.append(chunk_data.isel({self.timedim: relative_time_idx}).time.values)
                merge_child_ids.append(child_ids)
                merge_parent_ids.append(parent_ids)
                merge_areas.append(overlap_blobs_list[child_mask, 2])
                
                ### Relabel the Original Child Blob ID Field to account for the New ID:
                parent_centroids = blob_props.sel(ID=parent_ids).centroid.values.T  # (y, x), [:,0] are the y's
                
                if self.nn_partitioning:
                    # --> For every (Original) Child Cell in the ID Field, Find the closest (t-1) Parent _Cell_
                    parent_masks = np.zeros((len(parent_ids), blob_id_time.shape[0], blob_id_time.shape[1]), dtype=bool)
                    for idx, parent_id in enumerate(parent_ids):
                        parent_masks[idx] = (blob_id_time_m1 == parent_id).values
                    
                    # Calculate typical blob size to set max_distance
                    max_area = np.max(blob_props.sel(ID=parent_ids).area.values)
                    max_distance = int(np.sqrt(max_area) * 2.0)  # Use 2x the max blob radius
                    max_distance = max(max_distance, 20)  # Set minimum threshold...
                    
                    new_labels = get_nearest_parent_labels(
                        child_mask_2d,
                        parent_masks, 
                        child_ids,
                        parent_centroids,
                        Nx,
                        max_distance=max_distance
                    )
                else: 
                    # --> For every (Original) Child Cell in the ID Field, Find the closest (t-1) Parent _Centroid_
                    distances = wrapped_euclidian_parallel(child_mask_2d, parent_centroids, Nx)  # **Deals with wrapping**
                    
                    # Assign the new ID to each cell based on the closest parent
                    new_labels = child_ids[np.argmin(distances, axis=1)]
                
                
                ## Update values in child_time_idx and assign the updated slice back to the original DataArray
                temp = np.zeros_like(blob_id_time)
                temp[child_mask_2d] = new_labels
                blob_id_time = blob_id_time.where(~child_mask_2d, temp)
                ## ** Update directly into the chunk
                chunk_data[{self.timedim: relative_time_idx}] = blob_id_time
                
                
                ## Add new entries to time_index_map for each of new_blob_id corresponding to the current time index
                time_index_map.update({new_id: child_time_idx for new_id in new_blob_id})
                
                ## Update the Properties of the N Children Blobs
                new_child_props = self.calculate_blob_properties(blob_id_time, properties=['area', 'centroid'])
                
                # Update the blob_props DataArray:  (but first, check if the original Children still exists)
                if child_id in new_child_props.ID:  # Update the entry
                    blob_props.loc[dict(ID=child_id)] = new_child_props.sel(ID=child_id)
                else:  # Delte child_id:  The blob has split/morphed such that it doesn't get a partition of this child...
                    blob_props = blob_props.drop_sel(ID=child_id)  # N.B.: This means that the IDs are no longer continuous...
                    print(f"Deleted child_id {child_id} because parents have split/morphed in the meantime...")
                # Add the properties for the N-1 other new child ID
                new_blob_ids_still = new_child_props.ID.where(new_child_props.ID.isin(new_blob_id), drop=True).ID
                blob_props = xr.concat([blob_props, new_child_props.sel(ID=new_blob_ids_still)], dim='ID')
                missing_ids = set(new_blob_id) - set(new_blob_ids_still.values)
                if len(missing_ids) > 0:
                    print(f"Missing newly created child_ids {missing_ids} because parents have split/morphed in the meantime...")

                
                ## Finally, Re-assess all of the Parent IDs (LHS) equal to the (original) child_id
                
                # Look at the overlap IDs between the original child_id and the next time-step, and also the new_blob_id and the next time-step
                new_overlaps = self.check_overlap_slice(blob_id_time.values, blob_id_time_p1.values)
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
                
                
                ## Finally, _FINALLY_, we need to ensure that of the new children blobs we made, they only overlap with their respective parent...
                new_unique_children, new_children_counts = np.unique(new_child_overlaps_list[:, 1], return_counts=True)
                new_merging_blobs = new_unique_children[new_children_counts > 1]
                if new_merging_blobs.size > 0:
                    
                    if relative_time_idx + 1 < chunk_data.sizes[self.timedim]-1:  # If there is a next time-step in this chunk
                        for new_child_id in new_merging_blobs:
                            if new_child_id not in blobs_to_process: # We aren't already going to assess this blob
                                blobs_to_process.insert(0, new_child_id)
                    
                    else: # This is out of our current jurisdiction: Defer this reassessment to the beginning of the next chunk
                        future_chunk_merges.extend(new_merging_blobs)
                
            
            # Update the full dask DataArray with this processed chunk
            blob_id_field_unique[{
                self.timedim: slice(chunk_start, chunk_end-1)  # cf. above definition of chunk_end for why we need -1
            }] = chunk_data[:(chunk_end-1-chunk_start)]
            
            if chunk_idx % 25 == 0:
                print(f"Processing splitting and merging in chunk {chunk_idx} of {len(blobs_by_chunk)}")
        
        ### Process the Merge Events
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
        
        
        blob_id_field_unique = blob_id_field_unique.persist()
        
        return (blob_id_field_unique, 
                blob_props, 
                overlap_blobs_list[:, :2],
                merge_events)
        
    
    
    def track_blObs(self, data_bin):
        '''Identifies Blobs across time, accounting for splitting & merging logic.
        
        Returns
        -------
        blob_id_field : xarray.DataArray
            Field of globally unique integer IDs of each element in connected regions. ID = 0 indicates no object.
        '''
        
        # Cluster & ID Binary Data at each Time Step
        blob_id_field, _ = self.identify_blobs(data_bin, time_connectivity=False)
        print('Finished blob identification.')
        
        # Calculate Properties of each Blob
        blob_props = self.calculate_blob_properties(blob_id_field, properties=['area', 'centroid'])
        print('Finished calculating blob properties.')
        
        # Compile List of Overlapping Blob ID Pairs Across Time
        overlap_blobs_list = self.find_overlapping_blobs(blob_id_field)  # List of overlapping blob pairs
        print('Finished finding overlapping blobs.')
        
        with warnings.catch_warnings():
            warnings.simplefilter("ignore", UserWarning)
            
            # Apply Splitting & Merging Logic to `overlap_blobs`
            #   N.B. This is the longest step due to loop-wise dependencies... but many sub-steps are highly threaded so we're okay-ish in the end
            split_merged_blob_id_field_unique, merged_blobs_props, split_merged_blobs_list, merge_events = self.split_and_merge_blobs(blob_id_field, blob_props, overlap_blobs_list)
            print('Finished splitting and merging blobs.')
            
            # Cluster Blobs List to Determine Globally Unique IDs & Update Blob ID Field
            split_merged_blobs_ds = self.cluster_rename_blobs_and_props(split_merged_blob_id_field_unique, merged_blobs_props, split_merged_blobs_list, merge_events)
            print('Finished clustering and renaming blobs.')
        
        # Count Number of Blobs (This may have increased due to splitting)
        N_blobs = split_merged_blobs_ds.ID_field.max().compute().data
    
        return split_merged_blobs_ds, merge_events, N_blobs









##################################
### Optimised Helper Functions ###
##################################


@jit(nopython=True, parallel=True, fastmath=True)
def wrapped_euclidian_parallel(mask_values, parent_centroids_values, Nx):
    """
    Optimised function for computing wrapped Euclidean distances.
    
    Parameters:
    -----------
    mask_values : np.ndarray
        2D boolean array where True indicates points to calculate distances for
    parent_centroids_values : np.ndarray
        Array of shape (n_parents, 2) containing (y, x) coordinates of parent centroids
    Nx : int
        Size of the x-dimension for wrapping
        
    Returns:
    --------
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
    Creates a grid-based spatial index using numpy arrays.
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
    """
    dy = y1 - y2
    dx = x1 - x2
    
    if dx > half_nx:
        dx -= nx
    elif dx < -half_nx:
        dx += nx
        
    return np.sqrt(dy * dy + dx * dx)

@jit(nopython=True, parallel=True, fastmath=True)
def get_nearest_parent_labels(child_mask, parent_masks, child_ids, parent_centroids, Nx, max_distance=20):
    """
    Assigns labels based on nearest parent blob points.
    This is quite computationally-intensive, so we utilise many optimisations here...
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
            if found_close[child_idx]:  # Skip if we already found a very close match
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
                            
                        if dist < 2:  # Very close match found
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
                
                if min_dist_to_parent < 2:
                    found_close[child_idx] = True
    
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

