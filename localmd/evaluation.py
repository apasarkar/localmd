import jax
import jax.scipy
import jax.numpy as jnp
from jax import vmap
from typing import *
import numpy as np


def l1_norm(data: jnp.ndarray) -> jnp.ndarray:
    """
    Calculates the overall L1 norm of the data.

    Args:
        data (jnp.ndarray): The input data of any shape.

    Returns:
        (jnp.ndarray): The L1 norm of the input data.
    """

    data = jnp.abs(data)
    norm = jnp.sum(data)
    return norm


def trend_filter_stat(trace: jnp.ndarray) -> jnp.ndarray:
    """
    Applies a trend filter to a 1D time series dataset.

    Key assumption: Data has a length of at least 3.

    Args:
        trace (jnp.ndarray): The input time series data with shape (T,).

    Returns:
        (jnp.ndarray): The single value representing the trend filter statistic.
    """
    left_side = jax.lax.dynamic_slice(trace, (0,), (trace.shape[0] - 2,))
    right_side = jax.lax.dynamic_slice(trace, (2,), (trace.shape[0] - 2,))
    center = jax.lax.dynamic_slice(trace, (1,), (trace.shape[0] - 2,))

    combined_mat = center * 2 - left_side - right_side
    combined_mat = jnp.abs(combined_mat)
    return jnp.sum(combined_mat)


def total_variation_stat(img: jnp.ndarray) -> jnp.ndarray:
    """
    Applies a total variation filter to a 2D image.

    Key assumption: The image has a size of at least 3 x 3 pixels.

    Args:
        img (jnp.ndarray): The input image with shape (x, y).

    Returns:
        (jnp.ndarray): The total variation statistic for the input image.
    """
    center = jax.lax.dynamic_slice(img, (1, 1),
                                   (img.shape[0] - 2, img.shape[1] - 2))
    c00 = jax.lax.dynamic_slice(img, (0, 0),
                                (img.shape[0] - 2, img.shape[1] - 2))
    c10 = jax.lax.dynamic_slice(img, (1, 0),
                                (img.shape[0] - 2, img.shape[1] - 2))
    c20 = jax.lax.dynamic_slice(img, (2, 0),
                                (img.shape[0] - 2, img.shape[1] - 2))
    c21 = jax.lax.dynamic_slice(img, (2, 1),
                                (img.shape[0] - 2, img.shape[1] - 2))
    c22 = jax.lax.dynamic_slice(img, (2, 2),
                                (img.shape[0] - 2, img.shape[1] - 2))
    c12 = jax.lax.dynamic_slice(img, (1, 2),
                                (img.shape[0] - 2, img.shape[1] - 2))
    c02 = jax.lax.dynamic_slice(img, (0, 2),
                                (img.shape[0] - 2, img.shape[1] - 2))
    c01 = jax.lax.dynamic_slice(img, (0, 1),
                                (img.shape[0] - 2, img.shape[1] - 2))

    accumulator = jnp.zeros_like(center)

    accumulator = accumulator + jnp.abs(center - c00)
    accumulator = accumulator + jnp.abs(center - c10)
    accumulator = accumulator + jnp.abs(center - c20)
    accumulator = accumulator + jnp.abs(center - c21)
    accumulator = accumulator + jnp.abs(center - c22)
    accumulator = accumulator + jnp.abs(center - c12)
    accumulator = accumulator + jnp.abs(center - c02)
    accumulator = accumulator + jnp.abs(center - c01)

    return jnp.sum(accumulator)


def spatial_roughness_stat(u: jnp.ndarray) -> jnp.ndarray:
    """
    Computes a spatial roughness statistic for the input data

    Args:
        u (jnp.ndarray): Image of dimensions (d1, d2)

    Returns:
        (jnp.ndarray): The computed spatial roughness statistic
    """

    lower_vert = jax.lax.dynamic_slice(u, (1, 0), (u.shape[0] - 1, u.shape[1]))
    upper_vert = jax.lax.dynamic_slice(u, (0, 0), (u.shape[0] - 1, u.shape[1]))

    vert_diffs = jnp.abs(lower_vert - upper_vert)

    left_horz = jax.lax.dynamic_slice(u, (0, 0), (u.shape[0], u.shape[1] - 1))
    right_horz = jax.lax.dynamic_slice(u, (0, 1), (u.shape[0], u.shape[1] - 1))

    horz_diffs = jnp.abs(left_horz - right_horz)
    avg_diff = (jnp.sum(vert_diffs) + jnp.sum(horz_diffs)) / (
            vert_diffs.shape[0] * vert_diffs.shape[1] + horz_diffs.shape[0] * horz_diffs.shape[1])

    avg_elem = jnp.mean(jnp.abs(u))

    return avg_diff / avg_elem


def temporal_roughness_stat(v: jnp.ndarray) -> jnp.ndarray:
    """
    Args:
        v (jnp.ndarray): Input data of shape
    Returns:
        (jnp.ndarray): The computed temporal roughness statistic
    """

    v_left = jax.lax.dynamic_slice(v, (0,), (v.shape[0] - 2,))
    v_right = jax.lax.dynamic_slice(v, (2,), (v.shape[0] - 2,))
    v_middle = jax.lax.dynamic_slice(v, (1,), (v.shape[0] - 2,))

    return jnp.mean(jnp.abs(v_left + v_right - 2 * v_middle)) / jnp.mean(jnp.abs(v))


spatial_roughness_stat_vmap = vmap(spatial_roughness_stat, in_axes=(2))
temporal_roughness_stat_vmap = vmap(temporal_roughness_stat, in_axes=(0))


def evaluate_fitness(img: jnp.ndarray, trace: jnp.ndarray, spatial_threshold: Union[float, jnp.ndarray],
                     temporal_threshold: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """
    Evaluates the fitness of an image-trace pair based on spatial and temporal thresholds.

    Args:
        img (jnp.ndarray): The input image.
        trace (jnp.ndarray): The input trace data.
        spatial_threshold (Union[float, jnp.ndarray]): The threshold for spatial roughness.
            Can be a single float or jnp.ndarray value
        temporal_threshold (Union[float, jnp.ndarray]): The threshold for temporal roughness.
            Can be a single float or jnp.ndarray value

    Returns:
        jnp.ndarray: Output with dtype (int32) indicating the fitness evaluation. Returns 1 if both spatial and temporal
            conditions are met, otherwise returns 0.
    """
    spatial_stat = spatial_roughness_stat(img)
    temporal_stat = temporal_roughness_stat(trace)
    exp1 = spatial_stat < spatial_threshold
    exp2 = temporal_stat < temporal_threshold
    bool_exp = exp1 & exp2
    output = jax.lax.select(bool_exp, jnp.array([1]), jnp.array([0]))

    return output


evaluate_fitness_vmap = vmap(evaluate_fitness, in_axes=(2, 1, None, None))


def construct_final_fitness_decision(images: jnp.ndarray,
                                     traces: jnp.ndarray, spatial_threshold: Union[float, jnp.ndarray],
                                     temporal_threshold: Union[float, jnp.ndarray]) -> jnp.ndarray:
    """
    Constructs the final fitness decision for a batch of image-trace pairs based on spatial and temporal thresholds.

    Args:
        images (jnp.ndarray): An array containing input images.
        traces (jnp.ndarray): An array containing input trace data.
        spatial_threshold (Union[float, jnp.ndarray]): The threshold(s) for spatial roughness.
            Can be a single float or jnp.ndarray value.
        temporal_threshold (Union[float, jnp.ndarray]): The threshold(s) for temporal roughness.
            Can be a single float or jnp.ndarray value.

    Returns:
        jnp.ndarray: An array containing the fitness evaluation for each image-trace pair.
            Returns 1 if both spatial and temporal conditions are met for a pair, otherwise returns 0.
    """

    output = evaluate_fitness_vmap(images, traces, spatial_threshold, temporal_threshold)
    return output


def filter_by_failures(decisions: np.ndarray, max_consecutive_failures: int) -> np.ndarray:
    """
    Filters decisions based on maximum consecutive failures.

    Args:
        decisions (np.ndarray): 1-dimensional array of boolean values representing decisions.
        max_consecutive_failures (int): Maximum number of consecutive failures (ie decisions[i] == 0) allowed.

    Returns:
        np.ndarray: Filtered decisions with the same shape and type as input decisions.
    """
    number_of_failures = 0
    all_fails = False
    for k in range(decisions.shape[0]):
        if all_fails:
            decisions[k] = False
        elif not decisions[k]:
            number_of_failures += 1
            if number_of_failures == max_consecutive_failures:
                all_fails = True
        else:
            number_of_failures = 0
    return decisions
