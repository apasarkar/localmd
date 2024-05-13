import datetime
import math
import sys
import time
from functools import partial

import jax
import jax.numpy as jnp
import jax.scipy
import numpy as np
from scipy.sparse import coo_matrix, hstack, diags, csr_matrix
from jax import jit

from localmd.evaluation import spatial_roughness_stat_vmap, temporal_roughness_stat_vmap, \
    construct_final_fitness_decision, filter_by_failures
from localmd.pmd_loader import PMDLoader
from localmd.pmdarray import PMDArray

from typing import *


def display(msg):
    """
    Printing utility that logs time and flushes.
    """
    tag = '[' + datetime.datetime.today().strftime('%y-%m-%d %H:%M:%S') + ']: '
    sys.stdout.write(tag + msg + '\n')
    sys.stdout.flush()


@partial(jit)
def truncated_random_svd(input_matrix, key, rank_placeholder):
    '''
    Input: 
        - input_matrix. jnp.ndarray (d, T), where d is number of pixels, T is number of frames
        - key: jax pseudorandom key for random data gen
        - rank_placeholder: jax.ndarray with shape (rank). We use the shape (rank) to make a matrix with "rank" columns. This is
            a standard workaround for making sure this function can be jitted. 
    Key: This function assumes that (1) rank + num_oversamples is less than all dimensions of the input_matrix and (2) num_oversmples >= 1
    
    '''
    num_oversamples = 10
    rank = rank_placeholder.shape[0]
    d = input_matrix.shape[0]
    T = input_matrix.shape[1]
    random_data = jax.random.normal(key, (T, rank + num_oversamples))
    projected = jnp.matmul(input_matrix, random_data)
    Q, R = jnp.linalg.qr(projected)
    B = jnp.matmul(Q.T, input_matrix)
    U, s, V = jnp.linalg.svd(B, full_matrices=False)

    U_final = Q.dot(U)
    V = jnp.multiply(jnp.expand_dims(s, axis=1), V)

    #Final step: prune the rank 
    U_truncated = jax.lax.dynamic_slice(U_final, (0, 0), (U_final.shape[0], rank))
    V_truncated = jax.lax.dynamic_slice(V, (0, 0), (rank, V.shape[1]))
    return [U_truncated, V_truncated]


@partial(jit)
def decomposition_no_normalize_approx(block, key, rank_placeholder):
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1 * d2, T), order="F")
    decomposition = truncated_random_svd(block_2d, key, rank_placeholder)

    u_mat, v_mat = decomposition[0], decomposition[1]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order="F")
    spatial_statistics = spatial_roughness_stat_vmap(u_mat)
    temporal_statistics = temporal_roughness_stat_vmap(v_mat)

    return spatial_statistics, temporal_statistics


@partial(jit, static_argnums=(0, 1, 2))
def rank_simulation(d1, d2, T, rank_placeholder, key1, key2):
    noise_data = jax.random.normal(key1, (d1, d2, T))
    spatial, temporal = decomposition_no_normalize_approx(noise_data, key2, rank_placeholder)
    return spatial, temporal


def make_jax_random_key():
    ii32 = np.iinfo(np.int32)
    prng_input = np.random.randint(low=ii32.min, high=ii32.max, size=1, dtype=np.int32)[0]
    key = jax.random.PRNGKey(prng_input)

    return key


def threshold_heuristic(dimensions, num_comps=1, iters=250, percentile_threshold=5):
    spatial_list = []
    temporal_list = []

    d1, d2, T = dimensions
    rank_placeholder = np.zeros((num_comps,))
    for k in range(iters):
        key1 = make_jax_random_key()
        key2 = make_jax_random_key()
        x, y = rank_simulation(d1, d2, T, rank_placeholder, key1, key2)
        spatial_list.append(x)
        temporal_list.append(y)

    spatial_thres = np.percentile(np.array(spatial_list).flatten(), percentile_threshold)
    temporal_thres = np.percentile(np.array(temporal_list).flatten(), percentile_threshold)
    return spatial_thres, temporal_thres


@jit
def filter_and_decompose(block, mean_img, std_img, spatial_basis, projection_data, spatial_thres, temporal_thres,
                         max_consec_failures):
    '''
    Inputs: 
    block: jnp.ndarray. Dimensions (block_1, block_2, T). (block_1, block_2) are the dimensions of this patch of data, T is the number of frames.
    mean_img: jnp.ndarray. Dimensions (block_1, block_2). Mean image of this block (over entire dataset, not just the "T" frames this block contains). 
    std_img: jnp.ndarray. Dimensions (block_1, block_2). Nosie variance image of this block (over the entire dataset, not just the "T" frames this block contains). 
    spatial_basis: jnp.ndarray. Dimensions (block_1, block_2, svd_dim). Here, svd_dim is the dimension of the whole FOV svd we perform before doing the localized SVD on each spatial patch. 
    projection_data: jnp.ndarray. Dimensions (T, max_dimension). Used for the fast approximate SVD method.
    spatial_thres: float. Threshold used to determine whether an estimated spatial component from the SVD is just noise or contains useful signal
    temporal_thres: float. Threshold used to determine whether an estimated temporal component from the SVD is just noise or not.
    max_consec_failures: int, usually 1. After doing the truncated SVD, we iterate over the components, from most to least significant, and 
    '''

    ##Step 1: Standardize the data
    block -= mean_img[:, :, None]
    block /= std_img[:, :, None]

    return single_block_md(block, projection_data, spatial_thres, temporal_thres, max_consec_failures)


@partial(jit, static_argnums=(3,))
def single_block_md(block, key, rank_placeholder, temporal_avg_factor, spatial_thres, temporal_thres):
    '''
    Matrix Decomposition function for all blocks. 
    Key assumptions: 
    (1) number of frames in block is divisible by temporal_avg_factor
    (2) rank_placeholder is smaller than frames // temporal_avg_factor
    
    Inputs: 
        - block: jnp.array. Dimensions (block_1, block_2, T). (block_1, block_2) are the dimensions of this patch of data, T is the number of frames. We assume that this data has already been centered and noise-normalized
        - key: jax random number key. 
        - rank_placeholder: jnp.array. Dimensions (max_rank,). Maximum rank of the low-rank decomposition which we permit over this block. We pass this information via shape of an array to enable full JIT of this function 
        - spatial_thres. float. We compute a spatial roughness statistic for each spatial component to determine whether it is noise or smoother signal. This is the threshold for that test. 
        - temporal_thres. float. We compute a temporal roughness statistic for each temporal component to determine whether it is noise or smoother signal. This is the threshold for that test. 
        
    '''
    order = "F"
    d1, d2, T = block.shape

    block_2d = jnp.reshape(block, (d1 * d2, temporal_avg_factor, T // temporal_avg_factor), order=order)
    block_2d_avg = jnp.mean(block_2d, axis=1)

    # decomposition = truncated_random_svd(block_2d_avg, key, rank_placeholder)
    u_mat = truncated_random_svd(block_2d_avg, key, rank_placeholder)[0]
    v_mat = jnp.matmul(u_mat.T, jnp.reshape(block, (d1 * d2, T), order=order))
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order=order)

    # Now we begin the evaluation phase
    good_comps = construct_final_fitness_decision(u_mat, v_mat.T, spatial_thres, \
                                                  temporal_thres)

    return u_mat, good_comps, v_mat


@partial(jit, static_argnums=(4,))
def single_residual_block_md(block, existing, key, rank_placeholder, temporal_avg_factor, spatial_thres,
                             temporal_thres):
    '''
    Matrix Decomposition function for all blocks. 
    Inputs: 
        - block: jnp.array. Dimensions (block_1, block_2, T). (block_1, block_2) are the dimensions of this patch of data, T is the number of frames. We assume that this data has already been centered and noise-normalized
        - existing: jnp.array. Dimensions (block_1, block_2, T). (block_1, block_2) are the dimensions of this patch of data, T is the number of frames. This is an orthonormal spatial basis set which has already been identified for this spatial block of the FOV. We subtract it from "block" (via linear subspace projection) and THEN run the truncated SVD. The goal here is to find neural signal from "block" which is not already identified by "existing". 
        - key: jax random number key. 
        - rank_placeholder: jnp.array. Dimensions (max_rank,). Maximum rank of the low-rank decomposition which we permit over this block. We pass this information via shape of an array to enable full JIT of this function 
        - spatial_thres. float. We compute a spatial roughness statistic for each spatial component to determine whether it is noise or smoother signal. This is the threshold for that test. 
        - temporal_thres. float. We compute a temporal roughness statistic for each temporal component to determine whether it is noise or smoother signal. This is the threshold for that test. 
    '''
    order = "F"
    d1, d2, T = block.shape
    net_comps = existing.shape[2]
    block_2d = jnp.reshape(block, (d1 * d2, T), order=order)
    existing_2d = jnp.reshape(existing, (d1 * d2, net_comps), order=order)

    projection = jnp.matmul(existing_2d, jnp.matmul(existing_2d.transpose(), block_2d))
    block_2d = block_2d - projection

    block_r = jnp.reshape(block_2d, (d1 * d2, temporal_avg_factor, T // temporal_avg_factor), order=order)
    block_r_avg = jnp.mean(block_r, axis=1)

    u_mat = truncated_random_svd(block_r_avg, key, rank_placeholder)[0]
    v_mat = jnp.matmul(u_mat.T, jnp.reshape(block_2d, (d1 * d2, T), order=order))
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order=order)

    # Now we begin the evaluation phase
    good_comps = construct_final_fitness_decision(u_mat, v_mat.T, spatial_thres, \
                                                  temporal_thres)

    return u_mat, good_comps, v_mat


@partial(jit)
def get_temporal_projector(final_spatial_decomposition, block):
    '''
    Inputs: 
        final_spatial_decomposition. jnp.array. Shape (d1, d2, R), R is the rank. All columns orthonormal
        block: jnp.array. Shape (d1, d2, T), T is number of frames in data which we fit for PMD

    Returns: 
        temporal_decomposition: jnp.array. Shape (R, T). Projection of block onto spatial basis
    '''
    d1, d2, R = final_spatial_decomposition.shape
    T = block.shape[2]
    final_spatial_decomposition_r = jnp.reshape(final_spatial_decomposition, (d1 * d2, R), order="F")
    block_r = jnp.reshape(block, (d1 * d2, T), order="F")
    temporal_decomposition = jnp.matmul(final_spatial_decomposition_r.transpose(), block_r)
    return temporal_decomposition


def windowed_pmd(window_length, block, max_rank, spatial_thres, temporal_thres, max_consec_failures,
                 temporal_avg_factor):
    '''
    Implementation of windowed blockwise decomposition. Given a block of the movie (d1, d2, T), we break the movie into smaller chunks. 
    (say (d1, d2, R) where R < T), and run the truncated SVD decomposition iteratively on these blocks. This helps (1) avoid rank blowup and
    (2) make sure our spatial fit 
    
    Inputs: 
        - window_length: int. We break up the block into temporal subsets of this length and do the blockwise SVD decomposition on these blocks
        - block: np.ndarray. Shape (d1, d2, T)
        - max_rank: We break up "block" into temporal segments of length "window_length", and we run truncated SVD on each of these subsets iteratively. max_rank is the max rank of the decomposition we can obtain from any one of these individual blocks
        - spatial_thres: float. See single_block_md for docs
        - temporal_thres. float. See single_block_md for docs
        - max_consec_failures: int. After running the truncated SVD on this data, we look at each pair of rank-1 components (spatial, temporal) in order of significance (singular values). Once the hypothesis test fails a certain number of times on this data, we discard all subsequent components from the decomposition. 
    Returns: 
        - final_spatial_decomposition: np.ndarray. Shape (d1, d2, num_comps); this describes the spatial comps
        - final_temporal_decomposition: np.ndarray. Shape (num_comps, T); this describes the corresponding temporal comps
        Key: np.tensordot(final_spatial_decomposition, final_tempooral_decomposition, axes=(2,0)) should give the decomposition of the original data. 
    '''
    d1, d2 = (block.shape[0], block.shape[1])
    window_range = block.shape[2]
    assert window_length <= window_range
    start_points = list(range(0, window_range, window_length))

    final_spatial_decomposition = np.zeros((d1, d2, max_rank))
    remaining_components = max_rank

    component_counter = 0
    rank_placeholder = np.zeros((max_rank,))
    for k in start_points:
        start_value = k
        end_value = start_value + window_length

        key = make_jax_random_key()
        if k == 0 or component_counter == 0:
            subset = block[:, :, start_value:end_value]
            spatial_comps, decisions, _ = single_block_md(subset, key, rank_placeholder, temporal_avg_factor,
                                                          spatial_thres, temporal_thres)
        else:
            subset = block[:, :, start_value:end_value]
            spatial_comps, decisions, _ = single_residual_block_md(subset, final_spatial_decomposition, key,
                                                                   rank_placeholder, temporal_avg_factor, spatial_thres,
                                                                   temporal_thres)

        spatial_comps = np.array(spatial_comps)
        decisions = np.array(decisions).flatten() > 0
        decisions = filter_by_failures(decisions, max_consec_failures)
        spatial_cropped = spatial_comps[:, :, decisions]
        final_filter_index = min(spatial_cropped.shape[2], remaining_components)
        spatial_cropped = spatial_cropped[:, :, :final_filter_index]

        final_spatial_decomposition[:, :,
        component_counter:component_counter + spatial_cropped.shape[2]] = spatial_cropped
        component_counter += spatial_cropped.shape[2]
        if component_counter == max_rank:
            break
        else:
            remaining_components = max_rank - component_counter

    #Run this first so that the jitted code can be reused across function calls (avoids recompilation...)
    final_temporal_decomposition = np.array(get_temporal_projector(final_spatial_decomposition, block))

    final_spatial_decomposition = final_spatial_decomposition[:, :, :component_counter]
    final_temporal_decomposition = final_temporal_decomposition[:component_counter, :]

    return final_spatial_decomposition, final_temporal_decomposition


def identify_window_chunks(frame_range, total_frames, window_chunks):
    '''
    Inputs: 
        frame_range: number of frames to fit
        total_frames: total number of frames in the movie
        window_chunks: we sample continuous chunks of data throughout the movie. each chunk is of size "window_chunks"
        
        Key requirements: 
        (1) frame_range should be less than total number of frames
        (2) window_chunks should be less than or equal to frame_range
    Returns:
        net_frames: list. Contains the starting point of the intervals (each of length "window_chunk") on which we do the decomposition.
    '''
    if frame_range > total_frames:
        raise ValueError("Requested more frames than available")
    if window_chunks > frame_range:
        raise ValueError("The size of each temporal chunk is bigger than frame range")

    num_itervals = math.ceil(frame_range / window_chunks)

    available_intervals = np.arange(0, total_frames, window_chunks)
    if available_intervals[-1] > total_frames - window_chunks:
        available_intervals[-1] = total_frames - window_chunks
    starting_points = np.random.choice(available_intervals, size=num_itervals, replace=False)
    starting_points = np.sort(starting_points)
    display("sampled from the following regions: {}".format(starting_points))

    net_frames = []
    for k in starting_points:
        curr_start = k
        curr_end = min(k + window_chunks, total_frames)

        curr_frame_list = [i for i in range(curr_start, curr_end)]
        net_frames.extend(curr_frame_list)
    return net_frames


def update_blocksizes(blocks: tuple, fov_shape: tuple, min_block_value: int = 10) -> tuple:
    """
    If user specifies block sizes that are too large, this approach truncates the blocksizes appropriately

    Args:
        blocks (tuple): Two integers, specifying the height and width blocksizes used in compression
        fov_shape (tuple): The height and width of the FOV
        min_block_value (int): The minimum value of a block in either spatial dimension.

    Returns:
        None

    Raises:
        ValueError if either block dimension is less than min allowed value.
    """
    if blocks[0] < min_block_value or blocks[1] < min_block_value:
        raise ValueError("One of the block dimensions was less than min allowed value of {}, "
                         "set to a larger value".format(min_block_value))
    final_blocks = []
    if blocks[0] > fov_shape[0]:
        display("Height blocksize was set to {} but corresponding dimension has size {}. Truncating to {}".format(
            blocks[0], fov_shape[0], fov_shape[0]
        ))
        final_blocks.append(fov_shape[0])
    else:
        final_blocks.append(blocks[0])
    if blocks[1] > fov_shape[1]:
        display("Height blocksize was set to {} but corresponding dimension has size {}. Truncating to {}".format(
            blocks[1], fov_shape[1], fov_shape[1]
        ))
        final_blocks.append(fov_shape[1])
    else:
        final_blocks.append(blocks[1])
    return final_blocks


def check_fov_size(fov_dims: tuple, min_allowed_value: int = 10) -> None:
    """
    Checks if the FOV dimensions are too small
    Args:
        fov_dims (tuple). Two integers specifying the FOV dimensions.

    Returns:
        None

    Raises:
        ValueError if either field of view dim is less than the minimum allowed value.
    """
    for k in fov_dims:
        if k < min_allowed_value:
            raise ValueError("At least one FOV dimension is lower than {}, "
                             "too small to process".format(min_allowed_value))


def localmd_decomposition(dataset_obj, block_sizes, frame_range, max_components=50, background_rank=15, sim_conf=5,
                          batching=10, frame_batch_size=10000, dtype='float32', num_workers=0, pixel_batch_size=5000,
                          registration_routine=None, max_consec_failures=1, rank_prune=False, temporal_avg_factor=10):
    check_fov_size((dataset_obj.shape[1], dataset_obj.shape[2]))
    load_obj = PMDLoader(dataset_obj, dtype=dtype, center=True, normalize=True, background_rank=background_rank,
                         batch_size=frame_batch_size, num_workers=num_workers, pixel_batch_size=pixel_batch_size,
                         registration_routine=registration_routine)

    #Decide which chunks of the data you will use for the spatial PMD blockwise fits
    window_chunks = 2000  #We will sample chunks of frames throughout the movie
    if load_obj.shape[0] <= frame_range:
        display("WARNING: Specified using more frames than there are in the dataset.")
        frame_range = load_obj.shape[0]
        start = 0
        end = load_obj.shape[0]
        frames = [i for i in range(start, end)]
        if frame_range <= window_chunks:
            display("WARNING: Initializing on less than {} frames, this will lead to limited benefits.".format(
                window_chunks))
            window_chunks = frame_range
    else:
        if frame_range <= window_chunks:
            if frame_range < window_chunks:
                display("WARNING: Initializing on less than {} frames, this will lead to limited benefits.".format(
                    window_chunks))
            window_chunks = frame_range
        frames = identify_window_chunks(frame_range, load_obj.shape[0], window_chunks)
    display("We are initializing on a total of {} frames".format(len(frames)))

    block_sizes = update_blocksizes(block_sizes, (dataset_obj.shape[1], dataset_obj.shape[2]))
    overlap = [math.ceil(block_sizes[0] / 2), math.ceil(block_sizes[1] / 2)]

    ##Get the spatial and temporal thresholds
    display(
        "Running Simulations, block dimensions are {} x {} x {} ".format(block_sizes[0], block_sizes[1], window_chunks))
    spatial_thres, temporal_thres = threshold_heuristic([block_sizes[0], block_sizes[1], window_chunks], num_comps=1,
                                                        iters=250, percentile_threshold=sim_conf)

    ##Load the data you will do blockwise SVD on
    display("Loading Data")
    data, temporal_basis_crop = load_obj.temporal_crop_with_filter(frames)

    ##Run PMD and get the compressed spatial representation of the data
    display("Obtaining blocks and running local SVD")

    dim_1_iters = list(range(0, data.shape[0] - block_sizes[0] + 1, block_sizes[0] - overlap[0]))
    if dim_1_iters[-1] != data.shape[0] - block_sizes[0] and data.shape[0] - block_sizes[0] != 0:
        dim_1_iters.append(data.shape[0] - block_sizes[0])

    dim_2_iters = list(range(0, data.shape[1] - block_sizes[1] + 1, block_sizes[1] - overlap[1]))
    if dim_2_iters[-1] != data.shape[1] - block_sizes[1] and data.shape[1] - block_sizes[1] != 0:
        dim_2_iters.append(data.shape[1] - block_sizes[1])

    #Define the block weighting matrix
    block_weights = np.ones((block_sizes[0], block_sizes[1]), dtype=dtype)
    hbh = block_sizes[0] // 2
    hbw = block_sizes[1] // 2
    # Increase weights to value block centers more than edges
    block_weights[:hbh, :hbw] += np.minimum(
        np.tile(np.arange(0, hbw), (hbh, 1)),
        np.tile(np.arange(0, hbh), (hbw, 1)).T
    )
    block_weights[:hbh, hbw:] = np.fliplr(block_weights[:hbh, :hbw])
    block_weights[hbh:, :] = np.flipud(block_weights[:hbh, :])

    sparse_indices = np.arange(data.shape[0] * data.shape[1]).reshape((data.shape[0], data.shape[1]),
                                                                      order=load_obj.order)
    row_number = 0
    column_indices = []
    row_indices = []
    spatial_overall_values = []
    cumulative_weights = np.zeros((data.shape[0], data.shape[1]))
    total_temporal_fit = []

    if temporal_avg_factor >= data.shape[2]:
        raise ValueError("Need at least {} frames".format(temporal_avg_factor))
    if data.shape[2] // temporal_avg_factor <= max_components:
        string_to_disp = (
            f"WARNING: temporal avg factor is too big, max rank per block adjusted to {data.shape[2] // temporal_avg_factor}.\n"
            "To avoid this, initialize with more frames or reduce temporal avg factor")
        display(string_to_disp)
        max_components = int(data.shape[2] // temporal_avg_factor)

    #Key: Crop temporal_basis_crop here. Long term refactor this
    crop_avg_constant = (data.shape[2] // temporal_avg_factor) * temporal_avg_factor
    temporal_basis_crop = temporal_basis_crop[:, :crop_avg_constant]

    pairs = []
    for k in dim_1_iters:
        for j in dim_2_iters:
            pairs.append((k, j))
            subset = data[k:k + block_sizes[0], j:j + block_sizes[1], :].astype(dtype)
            subset = subset[:, :, :crop_avg_constant]
            spatial_cropped, temporal_cropped = windowed_pmd(window_chunks, subset, max_components, spatial_thres,
                                                             temporal_thres, max_consec_failures, temporal_avg_factor)
            total_temporal_fit.append(temporal_cropped)

            #Weight the spatial components here
            spatial_cropped = spatial_cropped * block_weights[:, :, None]
            current_cumulative_weight = block_weights
            cumulative_weights[k:k + block_sizes[0], j:j + block_sizes[1]] += current_cumulative_weight

            sparse_col_indices = sparse_indices[k:k + block_sizes[0], j:j + block_sizes[1]][:, :, None]
            sparse_col_indices = sparse_col_indices + np.zeros((1, 1, spatial_cropped.shape[2]))
            sparse_row_indices = np.zeros_like(sparse_col_indices)
            addend = np.arange(row_number, row_number + spatial_cropped.shape[2])[None, None, :]

            sparse_row_indices = sparse_row_indices + addend
            sparse_col_indices_f = sparse_col_indices.flatten().tolist()
            sparse_row_indices_f = sparse_row_indices.flatten().tolist()
            spatial_values_f = spatial_cropped.flatten().tolist()

            column_indices.extend(sparse_col_indices_f)
            row_indices.extend(sparse_row_indices_f)
            spatial_overall_values.extend(spatial_values_f)
            row_number += spatial_cropped.shape[2]

    U_r = coo_matrix((spatial_overall_values, (column_indices, row_indices)),
                     shape=(data.shape[0] * data.shape[1], row_number))
    V_cropped = np.concatenate(total_temporal_fit, axis=0)

    display("Normalizing by weights")
    weight_normalization_diag = np.zeros((data.shape[0] * data.shape[1],))
    weight_normalization_diag[sparse_indices.flatten()] = cumulative_weights.flatten()
    normalizing_weights = diags(
        [(1 / weight_normalization_diag).ravel()], [0])
    U_r = normalizing_weights.dot(U_r)

    U_r, V_cropped = aggregate_uv(U_r, V_cropped, load_obj.spatial_basis, temporal_basis_crop)
    display("The total rank before pruning is {}".format(U_r.shape[1]))

    display("Performing rank pruning and orthogonalization for fast sparse regression.")
    if rank_prune:
        U_r, P = get_projector(U_r, V_cropped)
    else:
        U_r, P = get_projector_noprune(U_r)
    display("After performing rank reduction, the updated rank is {}".format(P.shape[1]))

    ## Step 2f: Do sparse regression to get the V matrix: 
    display("Running sparse regression")
    V = load_obj.V_projection([U_r.T, P.T])

    #Extract necessary info from the loader object and delete it. This frees up space on GPU for the below linalg.eigh computations
    std_img = load_obj.std_img
    mean_img = load_obj.mean_img
    order = load_obj.order
    shape = load_obj.shape
    del load_obj

    ## Step 2h: Do a SVD Reformat given U and V
    display("Final reformat of data into complete SVD")
    R, s, Vt = factored_svd(P, V)
    R = np.array(R)
    s = np.array(s)
    Vt = np.array(Vt)

    display("Matrix decomposition completed")

    final_movie = PMDArray(U_r, R, s, Vt, shape, order, mean_img, std_img)
    return final_movie


def aggregate_uv(u: coo_matrix, v: np.ndarray, spatial_basis: np.ndarray,
                 temporal_basis: np.ndarray) -> Tuple[coo_matrix, np.ndarray]:
    """
    Aggregates the input matrices U and V with additional spatial and temporal bases.

    Args:
        u (scipy.sparse.coo_matrix): Input matrix U. Shape (d, R).
        v (np.ndarray): Input matrix V. Shape (R, T).
        spatial_basis (np.ndarray): Spatial basis matrix. Shape (d, K).
        temporal_basis (np.ndarray): Temporal basis matrix. Shape (K, T).

    Returns:
        tuple: A tuple containing:
            - coo_matrix: Aggregated U matrix. Shape (d, R + K).
            - np.ndarray: Aggregated V matrix. Shape (R + K, T).
    """
    spatial_bg_sparse = coo_matrix(spatial_basis)
    u_net = hstack([u, spatial_bg_sparse])

    v_net = np.concatenate([v, temporal_basis], axis=0)
    return u_net, v_net


def get_projector_noprune(U, tol=0.0001):
    UtU = U.T.dot(U)
    random_mat = np.eye(U.shape[1])
    UtUR = UtU.dot(random_mat)
    RtUtUR = np.array(jnp.matmul(random_mat.T, UtUR))

    eig_vecs, eig_vals, _ = jnp.linalg.svd(RtUtUR, full_matrices=False, hermitian=True)
    eig_vals = np.array(eig_vals)
    eig_vecs = np.array(eig_vecs)

    #Now filter any remaining bad components
    good_components = np.logical_and(np.abs(eig_vals) > tol, eig_vals > 0)

    #Apply the eigenvectors to random_mat
    random_mat_e = np.array(jnp.matmul(random_mat, eig_vecs))

    singular_values = np.sqrt(eig_vals)

    random_mat_e = random_mat_e[:, good_components]
    singular_values = singular_values[good_components]
    random_mat_e = random_mat_e / singular_values[None, :]

    return (U, random_mat_e)


def get_projector(U, V, rank_prune_target: float = 3, deterministic: bool = False):
    '''
    This function uses random projection method described in Halko to find an orthonormal subspace which approximates the 
    column span of UV. We want to express this subspace as a factorization: UP; this way we can keep the nice sparsity and avoid ever dealing with dense d x K (for any K) matrices (where d = number of pixels in movie). 
    
    Due to the overcomplete blockwise decomposition of PMD, we want to prune the rank of the PMD decomposition (U) by a factor of 4. We do this before regressing the entire movie onto the PMD object for memory and computational purposes (faster regression, more efficient GPU utilization). 
    Input: 
        U: scipy.sparse matrix of dimensions (d, R) where d is number of pixels, R is number of frames
    Returns: 
        Tuple (U, P) (described above)
    '''
    rank_prune_factor = rank_prune_target / 1.05
    tol = 0.0001
    keep_value = min(int(U.shape[1] / rank_prune_target), V.shape[1])

    if not deterministic:
        if int(U.shape[1] / rank_prune_target) < V.shape[1] and rank_prune_target > 1:
            random_mat = np.random.randn(V.shape[1], int(U.shape[1] / rank_prune_factor))
            random_mat = np.array(jnp.matmul(V, random_mat))
        else:
            random_mat = V
        UtU = U.T.dot(U)
        UtUR = UtU.dot(random_mat)
        RtUtUR = np.array(jnp.matmul(random_mat.T, UtUR))

        eig_vecs, eig_vals, _ = jnp.linalg.svd(RtUtUR, full_matrices=False, hermitian=True)
        eig_vals = np.array(eig_vals)
        eig_vecs = np.array(eig_vecs)

        #Now filter any remaining bad components
        good_components = np.logical_and(np.abs(eig_vals) > tol, eig_vals > 0)

        #Apply the eigenvectors to random_mat
        random_mat_e = np.array(jnp.matmul(random_mat, eig_vecs))

        singular_values = np.sqrt(eig_vals)

        random_mat_e = random_mat_e / singular_values[None, :]

        random_mat_e = random_mat_e[:, good_components]
        random_mat_e = random_mat_e[:, :keep_value]
        return (U, random_mat_e)

    else:
        display("For Reference purposes only: DETERMINISTIC")
        R, s, T = factored_svd_debug(U, V, factor=0.5)
        return U, R


def eigenvalue_and_eigenvector_routine(sigma):
    eig_vals, eig_vecs = jnp.linalg.eigh(sigma)  # Note: eig vals/vecs ascending
    eig_vals = np.array(eig_vals)
    eig_vecs = np.array(eig_vecs)
    eig_vecs = np.flip(eig_vecs, axis=(1,))
    eig_vals = np.flip(eig_vals, axis=(0,))

    return eig_vecs, eig_vals


def compute_sigma(spatial_components, Lt):
    '''
    Note: Lt here refers to the upper triangular matrix from the QR factorization of V ("temporal_components"). So 
    temporal_components.T = Qt.dot(Lt), which means that 
    UV = U(Lt.T)(Qt.T)
    '''
    Lt = np.array(Lt)
    UtU = spatial_components.T.dot(spatial_components)
    UtUL = UtU.dot(Lt.T)
    Sigma = Lt.dot(UtUL)

    return Sigma


def rank_prune_svd(mixing_weights, singular_values, temporal_basis, factor=0.25):
    '''
    Inputs: 
        mixing_weights: numpy.ndarray, shape (R x R)
        singular_values: numpy.ndarray, shape (R)
        temporal_basis: numpy.ndarray, shape (R, T)
    '''
    dimension = singular_values.shape[0]
    index = int(math.floor(factor * dimension))
    if index == 0:
        pass
    elif index > singular_values.shape[0]:
        pass
    else:
        mixing_weights = mixing_weights[:, :index]
        singular_values = singular_values[:index]
        temporal_basis = temporal_basis[:index, :]
    display("The rank was originally {} now it is {}".format(dimension, mixing_weights.shape[1]))
    return mixing_weights, singular_values, temporal_basis


def factored_svd_debug(spatial_components, temporal_components, factor=0.25):
    '''
    This is a fast method to convert a low-rank decomposition (spatial_components * temporal_components) into a 
    Inputs: 
        spatial_components: scipy.sparse.coo_matrix. Shape (d, R)
        temporal_components: jax.numpy. Shape (R, T)
    '''
    Qt, Lt = jnp.linalg.qr(temporal_components.transpose(), mode='reduced')
    Sigma = compute_sigma(spatial_components, Lt)
    eig_vecs, eig_vals = eigenvalue_and_eigenvector_routine(Sigma)
    Qt = np.array(Qt)
    Lt = np.array(Lt)

    eig_vec_norms = np.linalg.norm(eig_vecs, axis=0)
    selected_indices = eig_vals > 0
    eig_vecs = eig_vecs[:, selected_indices]
    eig_vals = eig_vals[selected_indices]
    singular_values = np.sqrt(eig_vals)

    mixing_weights = np.array(jnp.matmul(Lt.T, eig_vecs / singular_values[None, :]))
    temporal_basis = np.array(jnp.matmul(eig_vecs.T, Qt.T))

    #Here we prune the factorized SVD 
    return rank_prune_svd(mixing_weights, singular_values, temporal_basis, factor=factor)


def factored_svd(projection: Union[np.ndarray, jnp.ndarray],
                 data: Union[np.ndarray, jnp.ndarray]) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
     Computes Singular Value Decomposition (SVD) of the input data and applies projection accordingly.

     Args:
         projection (Union[np.ndarray, jnp.ndarray]): Projection matrix which multiplies the SVD expression from the
         left or right.
         data (Union[np.ndarray, jnp.ndarray]): Input data for SVD computation.

     Returns:
         tuple: A tuple containing:
             - jnp.ndarray: Result of the projection applied to the singular vectors.
             - jnp.ndarray: Singular values obtained from the SVD.
             - jnp.ndarray: Right singular vector matrix.
     """
    d1, d2 = data.shape
    if d1 <= d2:
        display("Short matrix, using leftward SVD routine")
        return more_rows_svd_routine(projection, data)
    else:
        display("Tall matrix, using rightward SVD routine")
        return more_columns_svd_routine(projection, data)


@partial(jit)
def more_rows_svd_routine(projection: jnp.ndarray, data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Performs Singular Value Decomposition (SVD) on data routine with additional projection.

    Args:
        projection (jnp.ndarray): Projection matrix which multiplies the SVD expression from the left.
        data (jnp.ndarray): (x, y)-shaped data, where x > y.

    Returns:
        tuple: A tuple containing:
            - jnp.ndarray: Result of the projection applied to the left singular vectors.
            - jnp.ndarray: Singular values obtained from the SVD.
            - jnp.ndarray: Right singular vector matrix
    """
    v_vt = jnp.matmul(data, data.transpose())
    left, vals, _ = jnp.linalg.svd(v_vt, full_matrices=False, hermitian=True)
    singular_values = jnp.sqrt(vals)
    divisor = jnp.where(singular_values == 0, 1, singular_values)
    right_singular_matrix = jnp.divide(jnp.matmul(left.transpose(), data), jnp.expand_dims(divisor, 1))

    left_projection = jnp.matmul(projection, left)
    return left_projection, singular_values, right_singular_matrix


@partial(jit)
def more_columns_svd_routine(projection: jnp.ndarray,
                             data: jnp.ndarray) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Performs Singular Value Decomposition (SVD) on data routine with additional projection.

    Args:
        projection (jnp.ndarray): Projection matrix which multiplies the SVD expression from the left.
        data (jnp.ndarray): (x, y)-shaped data, where x < y.

    Returns:
        tuple: A tuple containing:
            - jnp.ndarray: Result of the projection applied to the left singular vectors.
            - jnp.ndarray: Singular values obtained from the SVD.
            - jnp.ndarray: Right singular vector matrix
    """
    vt_v = jnp.matmul(data.transpose(), data)
    right_t, vals, _ = jnp.linalg.svd(vt_v, full_matrices=False, hermitian=True)
    singular_values = jnp.sqrt(vals)
    divisor = jnp.where(singular_values == 0, 1, singular_values)

    left = jnp.matmul(data, jnp.divide(right_t, jnp.expand_dims(divisor, axis=0)))
    left_projection = jnp.matmul(projection, left)

    return left_projection, singular_values, right_t.transpose()
