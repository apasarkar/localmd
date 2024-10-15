import jax
import jax.scipy
import jax.numpy as jnp
import jax.numpy as jnp
from jax import jit, vmap
import functools
from functools import partial


@partial(jit)
def get_mean_and_noise(movie, mean_divisor):
    """
    This function is used to sum the movie chunk (and divide by total number of frames in ENTIRE movie, not the movie chunk) and to also calculate the noise variance in this chunk.
    """
    sum_val = jnp.sum(movie, axis=2) / mean_divisor
    d1, d2, T = movie.shape
    movie_centered_2d = jnp.reshape(movie, (d1 * d2, T), order="F")
    noise_estimate_1d = get_noise_estimate_vmap(movie_centered_2d)
    noise_estimate_2d = jnp.reshape(noise_estimate_1d, (d1, d2), order="F")
    return sum_val, noise_estimate_2d


@partial
def get_mean_chunk(movie, mean_divisor):
    return jnp.sum(movie, axis=2) / mean_divisor


def get_noise_estimate(trace):
    output_welch = jax.scipy.signal.welch(trace, noverlap=128)
    start = int(256 / 4 + 1)
    end = int(256 / 2 + 1)

    indices = jnp.arange(start, end)
    values = jnp.take(output_welch[1], indices) * 0.5
    sum_values = jnp.sum(values)

    return jnp.sqrt(sum_values / (end - start))


get_noise_estimate_vmap = vmap(get_noise_estimate, in_axes=(0))


@partial(jit)
def center_and_get_noise_estimate(movie, mean):
    """
    Goal of this function is to estimate the noise of a movie given the mean
    Input:
        movie: (j)np.ndarray. Dimensions (d1, d2, T), type float
        mean: (j)np.ndarray. Dimensions (d1, d2)
    """
    d1, d2, T = movie.shape
    movie_centered_3d = jnp.subtract(movie, jnp.expand_dims(mean, axis=2))
    movie_centered_2d = jnp.reshape(movie_centered_3d, (d1 * d2, T), order="F")
    noise_estimate_1d = get_noise_estimate_vmap(movie_centered_2d)
    noise_estimate_2d = jnp.reshape(noise_estimate_1d, (d1, d2), order="F")
    return noise_estimate_2d


@partial(jit)
def get_mean(trace):
    return jnp.mean(trace)


@partial(jit)
def center(trace):
    mean = get_mean(trace)
    return trace - mean


center_vmap = vmap(center, in_axes=(0))


@partial(jit)
def center_and_noise_normalize(trace):
    mean = get_mean(trace)
    centered_trace = trace - mean
    noise_est = get_noise_estimate(centered_trace)
    return centered_trace / noise_est


center_and_noise_normalize_vmap = jit(vmap(center_and_noise_normalize, in_axes=(0)))


@partial(jit)
def standardize_block(block):
    """
    Input:
        block: jnp.array. Dimensions (d1, d2, T)
    """
    d1, d2, T = block.shape
    block_2d = jnp.reshape(block, (d1 * d2, T), order="F")
    updated_2d = center_and_noise_normalize_vmap(block_2d)
    updated_3d = jnp.reshape(updated_2d, (d1, d2, T), order="F")
    return updated_3d
