import datetime
import math
import sys
from functools import partial

import jax
from jax import Array
from jax.typing import ArrayLike
import jax.numpy as jnp
import jax.scipy
import numpy as np
from scipy.sparse import coo_matrix, hstack, diags, csr_matrix
from jax import jit

from localmd.evaluation import (
    spatial_roughness_stat_vmap,
    temporal_roughness_stat_vmap,
    construct_final_fitness_decision,
    filter_by_failures,
)
from localmd.pmd_loader import PMDLoader
from localmd.pmdarray import PMDArray

from localmd.dataset import lazy_data_loader
from typing import *


def display(msg):
    """
    Printing utility that logs time and flushes.
    """
    tag = "[" + datetime.datetime.today().strftime("%y-%m-%d %H:%M:%S") + "]: "
    sys.stdout.write(tag + msg + "\n")
    sys.stdout.flush()


@partial(jit)
def truncated_random_svd(
    input_matrix: ArrayLike, key: ArrayLike, rank_placeholder: ArrayLike
) -> tuple[Array, Array]:
    """
    Runs a fast truncated SVD operation to get a low-rank truncated SVD of input_matrix. Uses randomness

    Args:
        input_matrix (ArrayLike): Shape (d, T), where d is number of pixels, T is number of frames
        key (ArrayLike) jax random key for random data gen
        rank_placeholder (ArrayLike): Shape (rank); used to make matrices with specific number of columns

    Returns:
        (tuple[Array, Array]): A tuple containing
            - (Array): Orthonormal truncated spatial basis
            - (Array): Temporal basis

    Note:
        This function assumes that (1) rank + num_oversamples is less than all dimensions of the input_matrix and
        (2) num_oversamples >= 1.
    """
    num_oversamples = 10
    rank = rank_placeholder.shape[0]
    t = input_matrix.shape[1]
    random_data = jax.random.normal(key, (t, rank + num_oversamples))
    projected = jnp.matmul(input_matrix, random_data)
    q, r = jnp.linalg.qr(projected)
    b = jnp.matmul(q.T, input_matrix)
    u, s, v = jnp.linalg.svd(b, full_matrices=False)

    u_final = q.dot(u)
    v = jnp.multiply(jnp.expand_dims(s, axis=1), v)

    # Final step: prune the rank
    u_truncated = jax.lax.dynamic_slice(u_final, (0, 0), (u_final.shape[0], rank))
    v_truncated = jax.lax.dynamic_slice(v, (0, 0), (rank, v.shape[1]))
    return u_truncated, v_truncated


@partial(jit)
def decomposition_no_normalize_approx(
    block: ArrayLike, key: ArrayLike, rank_placeholder: ArrayLike
) -> tuple[Array, Array]:
    """
    Runs the low rank decomposition pipeline without any normalization of pixels (centering, dividing by std dev, etc.)

    Args:
        block (ArrayLike): Shape (d1, d2, T)
        key (ArrayLike): jax random key used for random number gen
        rank_placeholder (ArrayLike): Shape (rank); used to make matrices with specific number of columns
    """
    order = "F"
    d1, d2, t = block.shape
    block_2d = jnp.reshape(block, (d1 * d2, t), order=order)
    decomposition = truncated_random_svd(block_2d, key, rank_placeholder)

    u_mat, v_mat = decomposition[0], decomposition[1]
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order=order)
    spatial_statistics = spatial_roughness_stat_vmap(u_mat)
    temporal_statistics = temporal_roughness_stat_vmap(v_mat)

    return spatial_statistics, temporal_statistics


@partial(jit, static_argnums=(0, 1, 2))
def rank_simulation(
    d1: int,
    d2: int,
    t: int,
    rank_placeholder: ArrayLike,
    key1: ArrayLike,
    key2: ArrayLike,
) -> tuple[Array, Array]:
    """
    Performs a simulation to compute the spatial and temporal roughness statistics of some data

    Args:
        d1 (int): The first spatial field of view dimension of the blocks we will decompose
        d2 (int): The second spatial field of view dimension of the blocks we will decompose
        t (int): The number of frames of data in the blocks we will decompose
        rank_placeholder (ArrayLike): Shape (rank); used to make matrices with specific number of columns
        key1 (ArrayLike): jax PRNG key
        key2 (ArrayLike): jax PRNG key

    Returns:
        tuple(Array, Array): A couple containing
            - Spatial statistic(s) of the simulated data
            - Temporal statistic(s) of the data
    """
    noise_data = jax.random.normal(key1, (d1, d2, t))
    spatial, temporal = decomposition_no_normalize_approx(
        noise_data, key2, rank_placeholder
    )
    return spatial, temporal


def make_jax_random_key() -> Array:
    """
    Returns a jax pseudorandom key
    """
    ii32 = np.iinfo(np.int32)
    prng_input = np.random.randint(low=ii32.min, high=ii32.max, size=1, dtype=np.int32)[
        0
    ]
    key = jax.random.PRNGKey(prng_input)

    return key


def threshold_heuristic(
    dimensions: tuple[int, int, int],
    num_comps: int = 1,
    iters: int = 250,
    percentile_threshold: float = 5,
) -> tuple[float, float]:
    """
    Generates a histogram of spatial and temporal roughness statistics from running the decomposition on random noise.
    This is used to decide how "smooth" the temporal and spatial components need to be in order to contain signal.

    Args:
        dimensions (tuple): Tuple describing the dimensions of the blocks which we will
            decompose. Contains (d1, d2, T), the two spatial field of view dimensions and the number of frames
        num_comps (int): The number of components which we identify in the decomposition
        iters (int): The number of times we run this simulation procedure to collect a histogram of spatial and temporal
            roughness statistics
        percentile_threshold (float): The threshold we use to decide whether the spatial and temporal roughness stats of
            decomposition are "smooth" enough to contain signal.

    Returns:
        tuple[float, float]: The spatial and temporal "cutoffs" for deciding whether a spatial-temporal decomposition
            contains signals.

    """
    spatial_list = []
    temporal_list = []

    d1, d2, t = dimensions
    rank_placeholder = np.zeros((num_comps,))
    for k in range(iters):
        key1 = make_jax_random_key()
        key2 = make_jax_random_key()
        x, y = rank_simulation(d1, d2, t, rank_placeholder, key1, key2)
        spatial_list.append(x)
        temporal_list.append(y)

    spatial_threshold = np.percentile(
        np.array(spatial_list).flatten(), percentile_threshold
    )
    temporal_threshold = np.percentile(
        np.array(temporal_list).flatten(), percentile_threshold
    )
    return spatial_threshold, temporal_threshold


@partial(jit, static_argnums=(3,))
def single_block_md(
    block: ArrayLike,
    key: ArrayLike,
    rank_placeholder: ArrayLike,
    temporal_avg_factor: int,
    spatial_threshold: float,
    temporal_threshold: float,
) -> tuple[Array, Array, Array]:
    """
    Runs the low rank truncated SVD decomposition on a subpatch of the data.
    Key assumptions:
    (1) number of frames in block is divisible by temporal_avg_factor
    (2) rank_placeholder is smaller than frames // temporal_avg_factor

    Args:
        block (ArrayLike): Dimensions (block_1, block_2, T).
            (block_1, block_2) are the dimensions of this patch of data, T is the number of frames.
                We assume all pixels have mean 0 and noise variance of 1 (data has been normalized)
        key (ArrayLike): jax PRNG key
        rank_placeholder (ArrayLike): Shape (rank); used to make matrices with specific number of columns
        temporal_avg_factor (int): We temporally average chunks frames of raw data to reduce noise; this parameter tells
            us how many frames are averaged together per "chunk"
        spatial_threshold (float): Threshold for deciding if a spatial component is smooth enough to contain signal
        temporal_threshold (float): Threshold for deciding if a temporal component is smooth enough to contain signal.

    Returns:
        tuple[Array, Array, Array]: The low-rank decomposition consisting of:
            - An orthogonal spatial basis of the data
            - A binary vector describing which components to keep based on the roughness statistic procedure
            - A temporal basis of the data
    """
    order = "F"
    d1, d2, t = block.shape

    block_2d = jnp.reshape(
        block, (d1 * d2, temporal_avg_factor, t // temporal_avg_factor), order=order
    )
    block_2d_avg = jnp.mean(block_2d, axis=1)

    # decomposition = truncated_random_svd(block_2d_avg, key, rank_placeholder)
    u_mat = truncated_random_svd(block_2d_avg, key, rank_placeholder)[0]
    v_mat = jnp.matmul(u_mat.T, jnp.reshape(block, (d1 * d2, t), order=order))
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order=order)

    # Now we begin the evaluation phase
    good_comps = construct_final_fitness_decision(
        u_mat, v_mat.T, spatial_threshold, temporal_threshold
    )

    return u_mat, good_comps, v_mat


@partial(jit, static_argnums=(4,))
def single_residual_block_md(
    block: ArrayLike,
    existing: ArrayLike,
    key: ArrayLike,
    rank_placeholder: ArrayLike,
    temporal_avg_factor: int,
    spatial_threshold,
    temporal_threshold,
):
    """
    Used to extract more components from a block of data in a low rank decomposition after running single_block_md

    Args:
        block (ArrayLike): Dimensions (block_1, block_2, T).
            (block_1, block_2) are the dimensions of this patch of data, T is the number of frames.
                We assume all pixels have mean 0 and noise variance of 1 (data has been normalized)
        existing (ArrayLike): Orthogonal spatial basis of existing decomposition; shape (block_1, block_2, rank)
        key: jax random number key.
        rank_placeholder (ArrayLike): Shape (rank); used to make matrices with specific number of columns
        temporal_avg_factor (int): We temporally average chunks frames of raw data to reduce noise; this parameter tells
            us how many frames are averaged together per "chunk"
        spatial_threshold (float): Threshold for deciding if a spatial component is smooth enough to contain signal
        temporal_threshold (float): Threshold for deciding if a temporal component is smooth enough to contain signal.

    Returns:
        tuple[Array, Array, Array]: The low-rank decomposition consisting of:
            - An orthogonal spatial basis of the data
            - A binary vector describing which components to keep based on the roughness statistic procedure
            - A temporal basis of the data
    """
    order = "F"
    d1, d2, t = block.shape
    net_comps = existing.shape[2]
    block_2d = jnp.reshape(block, (d1 * d2, t), order=order)
    existing_2d = jnp.reshape(existing, (d1 * d2, net_comps), order=order)

    projection = jnp.matmul(existing_2d, jnp.matmul(existing_2d.transpose(), block_2d))
    block_2d = block_2d - projection

    block_r = jnp.reshape(
        block_2d, (d1 * d2, temporal_avg_factor, t // temporal_avg_factor), order=order
    )
    block_r_avg = jnp.mean(block_r, axis=1)

    u_mat = truncated_random_svd(block_r_avg, key, rank_placeholder)[0]
    v_mat = jnp.matmul(u_mat.T, jnp.reshape(block_2d, (d1 * d2, t), order=order))
    u_mat = jnp.reshape(u_mat, (d1, d2, u_mat.shape[1]), order=order)

    # Now we begin the evaluation phase
    good_comps = construct_final_fitness_decision(
        u_mat, v_mat.T, spatial_threshold, temporal_threshold
    )

    return u_mat, good_comps, v_mat


@partial(jit)
def get_temporal_projector(spatial_decomposition: ArrayLike, block: ArrayLike) -> Array:
    """
    Args:
        spatial_decomposition (ArrayLike): Shape (d1, d2, r), r is the rank. All columns orthonormal
        block (ArrayLike): Shape (d1, d2, t), t is number of frames in data which we fit for PMD

    Returns:
        temporal_decomposition (Array): Shape (r, t). Projection of block onto spatial basis
    """
    d1, d2, r = spatial_decomposition.shape
    t = block.shape[2]
    spatial_decomposition_r = jnp.reshape(
        spatial_decomposition, (d1 * d2, r), order="F"
    )
    block_r = jnp.reshape(block, (d1 * d2, t), order="F")
    temporal_decomposition = jnp.matmul(spatial_decomposition_r.transpose(), block_r)
    return temporal_decomposition


def windowed_pmd(
    window_length: int,
    block: ArrayLike,
    max_rank: int,
    spatial_threshold: float,
    temporal_threshold: float,
    max_consecutive_failures: int,
    temporal_avg_factor: int,
) -> tuple[np.ndarray, np.ndarray]:
    """
    Implementation of windowed blockwise decomposition. Given a block of the movie (d1, d2, T), we break the movie
    into smaller chunks. (say (d1, d2, R) where R < T), and run the truncated SVD decomposition iteratively on these
    blocks. This helps (1) avoid rank blowup and (2) make sure our spatial fit captures signal properly

    Args:
        window_length (int). We break up the block into temporal subsets of this length and do the blockwise SVD
            decomposition on these blocks
        block (ArrayLike): Shape (d1, d2, T). A chunk of "T" frames of the input data used to get the decomposition
        max_rank (int): We break up "block" into temporal segments of length "window_length", and we run truncated SVD
            on each of these subsets iteratively. max_rank is the max rank of the decomposition we can obtain from any
            one of these individual blocks
        spatial_threshold (float): Spatial roughness statistic cut off; See single_block_md for details
        temporal_threshold (float): Temporal roughness statistic cut off; See single_block_md for docs
        max_consecutive_failures (int): After running the truncated SVD on this data, we look at each pair of rank-1
            components (spatial, temporal) in order of significance (singular values). Once the hypothesis test fails a
            certain number of times on this data, we discard all subsequent components from the decomposition.
        temporal_avg_factor (int): We temporally average chunks frames of raw data to reduce noise; this parameter tells
            us how many frames are averaged together per "chunk"
    Returns:
        final_spatial_decomposition (np.ndarray): Shape (d1, d2, num_comps); this describes the spatial basis
        final_temporal_decomposition (np.ndarray): Shape (num_comps, T); this describes the corresponding temporal comps

    Key: np.tensordot(final_spatial_decomposition, final_temporal_decomposition, axes=(2,0)) should give the
        decomposition of the input data
    """
    d1, d2 = (block.shape[0], block.shape[1])
    window_range = block.shape[2]
    if window_length > window_range:
        window_length = window_range
    start_points = list(range(0, window_range, window_length))

    # This guarantees that we always load a large number of frames when doing the decompositions below
    if len(start_points) > 0 and start_points[-1] + window_length > window_range:
        start_points[-1] = window_range - window_length

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
            spatial_comps, decisions, _ = single_block_md(
                subset,
                key,
                rank_placeholder,
                temporal_avg_factor,
                spatial_threshold,
                temporal_threshold,
            )
        else:
            subset = block[:, :, start_value:end_value]
            spatial_comps, decisions, _ = single_residual_block_md(
                subset,
                final_spatial_decomposition,
                key,
                rank_placeholder,
                temporal_avg_factor,
                spatial_threshold,
                temporal_threshold,
            )

        spatial_comps = np.array(spatial_comps)
        decisions = np.array(decisions).flatten() > 0
        decisions = filter_by_failures(decisions, max_consecutive_failures)
        spatial_cropped = spatial_comps[:, :, decisions]
        final_filter_index = min(spatial_cropped.shape[2], remaining_components)
        spatial_cropped = spatial_cropped[:, :, :final_filter_index]

        final_spatial_decomposition[
            :, :, component_counter : component_counter + spatial_cropped.shape[2]
        ] = spatial_cropped
        component_counter += spatial_cropped.shape[2]
        if component_counter == max_rank:
            break
        else:
            remaining_components = max_rank - component_counter

    # Run this first so that the jitted code can be reused across function calls (avoids recompilation...)
    final_temporal_decomposition = np.array(
        get_temporal_projector(final_spatial_decomposition, block)
    )

    final_spatial_decomposition = final_spatial_decomposition[:, :, :component_counter]
    final_temporal_decomposition = final_temporal_decomposition[:component_counter, :]

    return final_spatial_decomposition, final_temporal_decomposition


def identify_window_chunks(
    frame_range: int, total_frames: int, window_chunks: int
) -> list:
    """
    Args:
        frame_range (int): Number of frames to fit
        total_frames (int): Total number of frames in the movie
        window_chunks (int): We sample continuous chunks of data throughout the movie.
            Each chunk is of size "window_chunks"

    Returns:
        (list): Contains the starting point of the intervals
            (each of length "window_chunk") on which we do the decomposition.

    Key requirements:
        (1) frame_range should be less than total number of frames
        (2) window_chunks should be less than or equal to frame_range
    """
    if frame_range > total_frames:
        raise ValueError("Requested more frames than available")
    if window_chunks > frame_range:
        raise ValueError("The size of each temporal chunk is bigger than frame range")

    num_intervals = math.ceil(frame_range / window_chunks)

    available_intervals = np.arange(0, total_frames, window_chunks)
    if available_intervals[-1] > total_frames - window_chunks:
        available_intervals[-1] = total_frames - window_chunks
    starting_points = np.random.choice(
        available_intervals, size=num_intervals, replace=False
    )
    starting_points = np.sort(starting_points)
    display("sampled from the following regions: {}".format(starting_points))

    net_frames = []
    for k in starting_points:
        curr_start = k
        curr_end = min(k + window_chunks, total_frames)

        curr_frame_list = [i for i in range(curr_start, curr_end)]
        net_frames.extend(curr_frame_list)
    return net_frames


def update_block_sizes(
    blocks: tuple, fov_shape: tuple, min_block_value: int = 10
) -> list:
    """
    If user specifies block sizes that are too large, this approach truncates the blocksizes appropriately

    Args:
        blocks (tuple): Two integers, specifying the height and width blocksizes used in compression
        fov_shape (tuple): The height and width of the FOV
        min_block_value (int): The minimum value of a block in either spatial dimension.

    Returns:
        list: A list containing the updated block sizes

    Raises:
        ValueError if either block dimension is less than min allowed value.
    """
    if blocks[0] < min_block_value or blocks[1] < min_block_value:
        raise ValueError(
            "One of the block dimensions was less than min allowed value of {}, "
            "set to a larger value".format(min_block_value)
        )
    final_blocks = []
    if blocks[0] > fov_shape[0]:
        display(
            "Height blocksize was set to {} but corresponding dimension has size {}. Truncating to {}".format(
                blocks[0], fov_shape[0], fov_shape[0]
            )
        )
        final_blocks.append(fov_shape[0])
    else:
        final_blocks.append(blocks[0])
    if blocks[1] > fov_shape[1]:
        display(
            "Height blocksize was set to {} but corresponding dimension has size {}. Truncating to {}".format(
                blocks[1], fov_shape[1], fov_shape[1]
            )
        )
        final_blocks.append(fov_shape[1])
    else:
        final_blocks.append(blocks[1])
    return final_blocks


def check_fov_size(fov_dims: Tuple[int, int], min_allowed_value: int = 10) -> None:
    """
    Checks if the field of view (FOV) dimensions are too small.

    Args:
        fov_dims (tuple): Two integers specifying the FOV dimensions.
        min_allowed_value (int, optional): The minimum allowed value for FOV dimensions. Defaults to 10.

    Returns:
        None

    Raises:
        ValueError: If either field of view dimension is less than the minimum allowed value.
    """
    for k in fov_dims:
        if k < min_allowed_value:
            raise ValueError(
                "At least one FOV dimension is lower than {}, "
                "too small to process".format(min_allowed_value)
            )


def localmd_decomposition(
    dataset_obj: lazy_data_loader,
    block_sizes: tuple,
    frame_range: int,
    max_components: int = 50,
    background_rank: int = 15,
    sim_conf: int = 5,
    frame_batch_size: int = 10000,
    dtype: str = "float32",
    num_workers: int = 0,
    pixel_batch_size: int = 5000,
    max_consecutive_failures=1,
    rank_prune: bool = False,
    rank_prune_factor: float = 0.33,
    temporal_avg_factor: int = 10,
    order: str = "F",
    window_chunks: Optional[int] = None,
    compute_normalizer: bool = True,
    pixel_weighting: Optional[np.ndarray] = None
):
    check_fov_size((dataset_obj.shape[1], dataset_obj.shape[2]))
    load_obj = PMDLoader(
        dataset_obj,
        dtype=dtype,
        background_rank=background_rank,
        batch_size=frame_batch_size,
        num_workers=num_workers,
        pixel_batch_size=pixel_batch_size,
        order=order,
        compute_normalizer = compute_normalizer
    )

    if window_chunks is None:
        window_chunks = frame_range
    # Decide which chunks of the data you will use for the spatial PMD blockwise fits
    if load_obj.shape[0] < frame_range:
        display("WARNING: Specified using more frames than there are in the dataset.")
        frame_range = load_obj.shape[0]
        start = 0
        end = load_obj.shape[0]
        frames = [i for i in range(start, end)]
        if frame_range <= window_chunks:
            window_chunks = frame_range
    else:
        if frame_range <= window_chunks:

            window_chunks = frame_range
        frames = identify_window_chunks(frame_range, load_obj.shape[0], window_chunks)
    display("We are initializing on a total of {} frames".format(len(frames)))

    block_sizes = update_block_sizes(
        block_sizes, (dataset_obj.shape[1], dataset_obj.shape[2])
    )
    overlap = [math.ceil(block_sizes[0] / 2), math.ceil(block_sizes[1] / 2)]

    ##Get the spatial and temporal thresholds
    display(
        "Running Simulations, block dimensions are {} x {} x {} ".format(
            block_sizes[0], block_sizes[1], window_chunks
        )
    )
    spatial_threshold, temporal_threshold = threshold_heuristic(
        [block_sizes[0], block_sizes[1], window_chunks],
        num_comps=1,
        iters=250,
        percentile_threshold=sim_conf,
    )

    ##Load the data you will do blockwise SVD on
    display("Loading Data")
    data, temporal_basis_crop = load_obj.temporal_crop_with_filter(frames)

    if pixel_weighting is not None:
        data *= pixel_weighting[:, :, None]

    ##Run PMD and get the compressed spatial representation of the data
    display("Obtaining blocks and running local SVD")

    dim_1_iters = list(
        range(0, data.shape[0] - block_sizes[0] + 1, block_sizes[0] - overlap[0])
    )
    if (
        dim_1_iters[-1] != data.shape[0] - block_sizes[0]
        and data.shape[0] - block_sizes[0] != 0
    ):
        dim_1_iters.append(data.shape[0] - block_sizes[0])

    dim_2_iters = list(
        range(0, data.shape[1] - block_sizes[1] + 1, block_sizes[1] - overlap[1])
    )
    if (
        dim_2_iters[-1] != data.shape[1] - block_sizes[1]
        and data.shape[1] - block_sizes[1] != 0
    ):
        dim_2_iters.append(data.shape[1] - block_sizes[1])

    # Define the block weighting matrix
    block_weights = np.ones((block_sizes[0], block_sizes[1]), dtype=dtype)
    hbh = block_sizes[0] // 2
    hbw = block_sizes[1] // 2
    # Increase weights to value block centers more than edges
    block_weights[:hbh, :hbw] += np.minimum(
        np.tile(np.arange(0, hbw), (hbh, 1)), np.tile(np.arange(0, hbh), (hbw, 1)).T
    )
    block_weights[:hbh, hbw:] = np.fliplr(block_weights[:hbh, :hbw])
    block_weights[hbh:, :] = np.flipud(block_weights[:hbh, :])

    sparse_indices = np.arange(data.shape[0] * data.shape[1]).reshape(
        (data.shape[0], data.shape[1]), order=load_obj.order
    )
    column_number = 0
    final_row_indices = []
    final_column_indices = []
    spatial_overall_values = []
    cumulative_weights = np.zeros((data.shape[0], data.shape[1]))
    total_temporal_fit = []

    if temporal_avg_factor >= data.shape[2]:
        raise ValueError("Need at least {} frames".format(temporal_avg_factor))
    if data.shape[2] // temporal_avg_factor <= max_components:
        string_to_disp = (
            f"WARNING: temporal avg factor is too big, max rank per block adjusted to {data.shape[2] // temporal_avg_factor}.\n"
            "To avoid this, initialize with more frames or reduce temporal avg factor"
        )
        display(string_to_disp)
        max_components = int(data.shape[2] // temporal_avg_factor)

    # Key: Crop temporal_basis_crop here. Long term refactor this
    crop_avg_constant = (data.shape[2] // temporal_avg_factor) * temporal_avg_factor
    temporal_basis_crop = temporal_basis_crop[:, :crop_avg_constant]

    pairs = []
    for k in dim_1_iters:
        for j in dim_2_iters:
            pairs.append((k, j))
            subset = data[k : k + block_sizes[0], j : j + block_sizes[1], :].astype(
                dtype
            )
            subset = subset[:, :, :crop_avg_constant]
            spatial_cropped, temporal_cropped = windowed_pmd(
                window_chunks,
                subset,
                max_components,
                spatial_threshold,
                temporal_threshold,
                max_consecutive_failures,
                temporal_avg_factor,
            )
            total_temporal_fit.append(temporal_cropped)

            # Weight the spatial components here
            spatial_cropped = spatial_cropped * block_weights[:, :, None]
            current_cumulative_weight = block_weights
            cumulative_weights[
                k : k + block_sizes[0], j : j + block_sizes[1]
            ] += current_cumulative_weight

            curr_spatial_row_indices = sparse_indices[
                k : k + block_sizes[0], j : j + block_sizes[1]
            ][:, :, None]
            curr_spatial_row_indices = curr_spatial_row_indices + np.zeros(
                (1, 1, spatial_cropped.shape[2])
            )

            curr_spatial_col_indices = np.zeros_like(curr_spatial_row_indices)
            addend = np.arange(column_number, column_number + spatial_cropped.shape[2])[
                None, None, :
            ]
            curr_spatial_col_indices = curr_spatial_col_indices + addend

            sparse_row_indices_f = curr_spatial_row_indices.flatten().tolist()
            sparse_col_indices_f = curr_spatial_col_indices.flatten().tolist()
            spatial_values_f = spatial_cropped.flatten().tolist()

            final_row_indices.extend(sparse_row_indices_f)
            final_column_indices.extend(sparse_col_indices_f)
            spatial_overall_values.extend(spatial_values_f)
            column_number += spatial_cropped.shape[2]

    u_r = coo_matrix(
        (spatial_overall_values, (final_row_indices, final_column_indices)),
        shape=(data.shape[0] * data.shape[1], column_number),
    )
    v_cropped = np.concatenate(total_temporal_fit, axis=0)

    display("Normalizing by weights")
    weight_normalization_diag = np.zeros((data.shape[0] * data.shape[1],))
    weight_normalization_diag[sparse_indices.flatten(order=load_obj.order)] = (
        cumulative_weights.flatten(order=load_obj.order)
    )

    normalizing_weights = diags([(1 / weight_normalization_diag).ravel()], [0])
    u_r = normalizing_weights.dot(u_r)

    u_r, v_cropped = aggregate_local_and_global_decomposition(
        u_r, v_cropped, load_obj.spatial_basis, temporal_basis_crop
    )
    display("The total rank before pruning is {}".format(u_r.shape[1]))

    display("Performing rank pruning and orthogonalization for fast sparse regression.")
    if rank_prune:

        if rank_prune_factor <= 0 or rank_prune_factor > 1:
            raise ValueError(
                "Rank prune factor should be a value in the interval (0, 1]"
            )

        min_dimension = min(u_r.shape[1], v_cropped.shape[1])

        random_key = make_jax_random_key()
        random_mat = jax.random.normal(
            random_key, (v_cropped.shape[1], int(min_dimension * rank_prune_factor))
        )
        temporal_mat_to_reformat = np.array(jnp.matmul(v_cropped, random_mat))
        p = compute_lowrank_factorized_svd(
            u_r, temporal_mat_to_reformat, only_left=True
        )
    else:
        p = compute_lowrank_factorized_svd(u_r, v_cropped, only_left=True)
    display(
        "After performing rank reduction, the updated rank is {}".format(p.shape[1])
    )

    ## Step 2f: Do sparse regression to get the v matrix:
    display("Running sparse regression")
    v = load_obj.v_projection(u_r, p)

    # Extract necessary info from the loader object and delete it. This frees up space on GPU for the below linalg.eigh computations
    std_img = load_obj.std_img
    mean_img = load_obj.mean_img
    order = load_obj.order
    shape = load_obj.shape
    del load_obj

    ## Step 2h: Do a SVD Reformat given u and v
    display("Final reformat of data into complete SVD")
    r, s, vt = projected_svd(p, v)
    r = np.array(r)
    s = np.array(s)
    vt = np.array(vt)

    display("Matrix decomposition completed")

    final_movie = PMDArray(u_r, r, s, vt, shape, order, mean_img, std_img)
    return final_movie


def aggregate_local_and_global_decomposition(
    u: coo_matrix, v: np.ndarray, spatial_basis: np.ndarray, temporal_basis: np.ndarray
) -> Tuple[coo_matrix, np.ndarray]:
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


def compute_lowrank_factorized_svd(
    u: coo_matrix, v: np.ndarray, only_left: bool = False
):
    """
    Compute the factorized Singular Value Decomposition (SVD) of a low-rank matrix factorization.

    This function computes the SVD of a matrix `u @ v`, where `u` is sparse and `v` is dense,
    both representing a low-rank factorization. It efficiently computes a reduced or partial SVD
    based on this factorization. The function allows returning just the left singular vectors
    (spatial mixing matrix) if specified.

    Args:
        u (scipy.sparse.coo_matrix):
            Sparse left matrix of the factorization with shape `(pixels, low_rank)`.
        v (np.ndarray):
            Dense right matrix of the factorization with shape `(low_rank, frames)`.
        only_left (bool, optional):
            If `True`, only the left singular vectors (spatial mixing matrix) are returned.
            Defaults to `False`.

    Returns:
        np.ndarray:
            `spatial_mixing_matrix`: An orthonormal column basis for the factorization `u @ v`.
            This matrix represents the spatial components of the original data.

        If `only_left` is False, it also returns:
        np.ndarray:
            `singular_values`: 1D vector of singular values, representing the scaling factors
            for the corresponding orthonormal directions.
        np.ndarray:
            `right_singular_vectors`: Orthonormal column vectors representing the temporal
            components of the matrix `v`.

    Notes:
        This is not a full SVD; the result is truncated to preserve efficiency, especially
        for large matrices. The orthogonality of the left singular vectors holds within the
        reduced space of the factorization.
    """
    ut_u = u.T.dot(u)

    if u.shape[1] > v.shape[1]:
        right_mat = v
    else:
        right_mat = np.eye(u.shape[1])

    ut_ur = ut_u.dot(right_mat)
    rtut_ur = np.array(jnp.matmul(right_mat.T, ut_ur))

    eig_vecs, eig_vals, _ = jnp.linalg.svd(rtut_ur, full_matrices=False, hermitian=True)
    eig_vals = np.array(eig_vals)
    eig_vecs = np.array(eig_vecs)

    good_components = eig_vals > 0
    eig_vecs = eig_vecs[:, good_components]
    eig_vals = eig_vals[good_components]

    # Apply the eigenvectors to random_mat
    spatial_mixing_matrix = np.array(jnp.matmul(right_mat, eig_vecs))

    singular_values = np.sqrt(eig_vals)
    spatial_mixing_matrix /= singular_values[None, :]

    if only_left:
        return spatial_mixing_matrix
    else:
        """
        Now we have u and spatial_mixing_matrix, which together form left singular vectors
        Our new factorization is (UP)(UP)^TUV = UP(P^TU^TUV). Remains to take the SVD of
        P^TU^TUV, and a lot of those computations are already done above (for e.g. U^TU)
        """
        new_temporal = jnp.matmul(spatial_mixing_matrix.T, (ut_u.dot(v)))
        spatial_mixing_matrix, singular_values, right_singular_vectors = projected_svd(
            spatial_mixing_matrix, new_temporal
        )
        return spatial_mixing_matrix, singular_values, right_singular_vectors


def projected_svd(
    projection: Union[np.ndarray, jnp.ndarray], data: Union[np.ndarray, jnp.ndarray]
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the Singular Value Decomposition (SVD) of the input `data` and applies a `projection` to the left singular vectors.

    This function performs two key operations:
    1. Computes the partial SVD of the input `data`, yielding the left singular vectors, singular values, and right singular vectors.
    2. Multiplies the `projection` matrix with the computed left singular vectors, resulting in a new set of left vectors, while retaining the singular values and right singular vectors.

    ### Application in PMD
    In the context of PMD, data is often factorized as `U @ P @ V`, where `U @ P` forms an orthonormal basis with `U` being sparse.
    To convert this factorization into a full singular value decomposition while preserving the sparsity of `U`, we apply this function.
    Specifically, calling `R, s, V_new = projected_svd(P, V)` produces the decomposition `(U @ R) * s * (V_new)`, which is the SVD of `U @ P @ V`.

    Args:
        projection (Union[np.ndarray, jnp.ndarray]): The projection matrix, typically `P`, applied to the left singular vectors after the SVD. Its shape should be compatible for multiplication with the left singular vectors obtained from `data`.
        data (Union[np.ndarray, jnp.ndarray]): The input data matrix on which the SVD is computed. It could be a dense or sparse matrix.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - **projected_left_vectors** (jnp.ndarray): The result of `projection @ left_vectors` from the partial SVD of `data`. These vectors represent the new left singular vectors after applying the projection.
            - **singular_values** (jnp.ndarray): The singular values from the SVD of `data`.
            - **right_singular_vectors** (jnp.ndarray): The right singular vectors from the SVD of `data`, representing the orthonormal basis for the column space of `data`.

    Notes:
        - This function is particularly useful in situations where a factorization needs to be converted to a full SVD, but maintaining the structure or sparsity of one factor is important.
        - The `projection` matrix allows applying transformations to the left singular vectors without modifying the singular values or right singular vectors.
    """
    d1, d2 = data.shape
    if d1 <= d2:
        display("Short matrix, using leftward SVD routine")
        left_singular_vectors, singular_values, right_singular_vectors = (
            fewer_rows_svd_routine(data)
        )
        left_singular_vectors = jnp.matmul(projection, left_singular_vectors)
    else:
        display("Tall matrix, using rightward SVD routine")
        left_singular_vectors, singular_values, right_singular_vectors = (
            fewer_columns_svd_routine(data)
        )
        left_singular_vectors = jnp.matmul(projection, left_singular_vectors)

    return left_singular_vectors, singular_values, right_singular_vectors


@partial(jit)
def fewer_rows_svd_routine(
    data: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the Singular Value Decomposition (SVD) for a matrix with more rows than columns

    This function leverages the fact that when a matrix has more rows than columns,
    it is computationally more efficient to perform the SVD on the smaller matrix
    (data @ data.T), rather than the original matrix.
    It returns the left singular vectors, singular values, and right singular vectors of the original matrix.

    Args:
        data (jnp.ndarray): A 2D input array with shape (x, y), where x (rows) > y (columns).
            This represents a matrix with more rows than columns, i.e., where the number
            of samples is larger than the number of features.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - jnp.ndarray: Left singular vectors, of shape (x, y). These form an orthonormal
              basis for the row space of the data.
            - jnp.ndarray: Singular values, of shape (y,). These are the square roots of the
              eigenvalues of (data @ data.T).
            - jnp.ndarray: Right singular vectors, of shape (y, y), representing the basis vectors
              for the original feature space.
    """
    v_vt = jnp.matmul(data, data.transpose())
    left_singular_vectors, vals, _ = jnp.linalg.svd(
        v_vt, full_matrices=False, hermitian=True
    )
    singular_values = jnp.sqrt(vals)
    divisor = jnp.where(singular_values == 0, 1, singular_values)
    right_singular_matrix = jnp.divide(
        jnp.matmul(left_singular_vectors.transpose(), data), jnp.expand_dims(divisor, 1)
    )

    return left_singular_vectors, singular_values, right_singular_matrix


@partial(jit)
def fewer_columns_svd_routine(
    data: jnp.ndarray,
) -> Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
    """
    Computes the Singular Value Decomposition (SVD) for a matrix with more columns than rows

    This function takes advantage of the fact that for matrices with more columns than rows,
    it is computationally cheaper to perform the SVD on the Gram matrix (data.T @ data)
    rather than the full matrix. It returns the left singular vectors, singular values,
    and right singular vectors of the original matrix.

    Args:
        data (jnp.ndarray): A 2D input array with shape (x, y), where x (rows) < y (columns).
            This represents a matrix with fewer rows than columns.

    Returns:
        Tuple[jnp.ndarray, jnp.ndarray, jnp.ndarray]:
            - jnp.ndarray: Left singular vectors, of shape (x, x). These are the projection
              of the input data onto the left singular basis.
            - jnp.ndarray: Singular values, of shape (x,). These are the square roots of the
              eigenvalues of (data.T @ data).
            - jnp.ndarray: Right singular vectors, of shape (y, x), which form the basis in
              the original column space of the data.
    """

    vt_v = jnp.matmul(data.transpose(), data)
    right_t, vals, _ = jnp.linalg.svd(vt_v, full_matrices=False, hermitian=True)
    singular_values = jnp.sqrt(vals)
    divisor = jnp.where(singular_values == 0, 1, singular_values)

    left_singular_vectors = jnp.matmul(
        data, jnp.divide(right_t, jnp.expand_dims(divisor, axis=0))
    )

    return left_singular_vectors, singular_values, right_t.transpose()
