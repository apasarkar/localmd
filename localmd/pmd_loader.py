import datetime
import math
import sys
from functools import partial
from sys import getsizeof

from scipy.sparse import coo_matrix
import jax
from jax import Array
from jax.typing import ArrayLike
from localmd.dataset import lazy_data_loader
import jax.numpy as jnp
import numpy as np
import torch
import torch.multiprocessing as multiprocessing
from jax import jit
from jax.experimental.sparse import BCOO
from tqdm import tqdm
from typing import *

from localmd.preprocessing_utils import get_mean_and_noise, get_mean_chunk


def display(msg):
    """
    Printing utility that logs time and flushes.
    """
    tag = "[" + datetime.datetime.today().strftime("%y-%m-%d %H:%M:%S") + "]: "
    sys.stdout.write(tag + msg + "\n")
    sys.stdout.flush()


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


@partial(jit, static_argnums=(2, 3))
def truncated_random_svd(
    input_matrix: ArrayLike, key: ArrayLike, rank: int, num_oversamples: int = 10
) -> tuple[ArrayLike, ArrayLike]:
    """
    Key: This function assumes that (1) rank + num_oversamples is less than all
        dimensions of the input_matrix and (2) num_oversmples >= 1
    """
    d = input_matrix.shape[0]
    T = input_matrix.shape[1]
    random_data = jax.random.normal(key, (T, rank + num_oversamples))
    projected = jnp.matmul(input_matrix, random_data)
    Q, R = jnp.linalg.qr(projected)
    B = jnp.matmul(Q.T, input_matrix)
    U, s, V = jnp.linalg.svd(B, full_matrices=False)

    U_final = Q.dot(U)
    V = jnp.multiply(jnp.expand_dims(s, axis=1), V)

    # Final step: prune the rank
    U_truncated = jax.lax.dynamic_slice(U_final, (0, 0), (U_final.shape[0], rank))
    V_truncated = jax.lax.dynamic_slice(V, (0, 0), (rank, V.shape[1]))
    return U_truncated, V_truncated


class FrameDataloader:
    def __init__(self, dataset: lazy_data_loader, batch_size: int, dtype="float32"):
        self.dataset = dataset
        self.chunks = math.ceil(self.shape[0] / batch_size)
        self.batch_size = batch_size
        self.dtype = dtype

    def __len__(self):
        return max(1, self.chunks - 1)

    @property
    def shape(self):
        return self.dataset.shape

    def __getitem__(self, index: int) -> np.ndarray:
        """
        Returns a chunk of frames
        Args:
            index (int): Which chunk of frames to be returned
        Returns:
            (np.ndarray): Shape (d1, d2, T); d1 and d2 are FOV dimensions and T is frames.
        """
        start = index * self.batch_size

        if index == max(0, self.chunks - 2):
            end = self.shape[0]
            # load rest of data here
            keys = [i for i in range(start, end)]
            data = self.dataset[keys].astype(self.dtype)
        elif index < self.chunks - 2:
            end = start + self.batch_size
            keys = [i for i in range(start, end)]
            data = self.dataset[keys].astype(self.dtype)
        else:
            raise ValueError

        # Data is shape (T, d1, d2), need to return (d1, d2, T)
        return data.transpose(1, 2, 0)


class PMDLoader:
    def __init__(
        self,
        dataset: lazy_data_loader,
        dtype="float32",
        background_rank: int = 15,
        batch_size: int = 2000,
        num_workers: int = None,
        pixel_batch_size: int = 5000,
        order: str = "F",
        compute_normalizer: bool = True,
    ):
        """
        Args:
            dataset: Object which implements the PMDDataset abstract interface. This is a basic reader which allows us
                to read frames of the input data.
            dtype: np.dtype. intended format of data
            background_rank: int. we run an approximate truncated svd on the full FOV of the data,
                of rank 'background_rank'. We subtract this from the data before
                running the core matrix decomposition compression method
            batch_size: max number of frames to load into memory (CPU and GPU) at a time
            num_workers: int, keep it at 0 for now. Number of workers used in pytorch dataloading.
                Experimental and best kept at 0.
            pixel_batch_size: int. maximum number of pixels of data we load onto GPU at any point in time
            order (str): "F" or "C" depending whether the 2D images in the video should be reshaped into a flattened,
                1D column vector in column major ("F") or row major ("C") order
            compute_normalizer (bool): Whether we compute a noise variance estimate per pixel in the normalizer
        """
        self._order = order
        self.dataset = dataset
        self.dtype = dtype

        self.shape = self.dataset.shape
        self.batch_size = batch_size
        self.pixel_batch_size = pixel_batch_size
        self._compute_normalizer = compute_normalizer

        def regular_collate(batch):
            return batch[0]

        self.curr_dataloader = FrameDataloader(
            self.dataset, self.batch_size, dtype=self.dtype
        )

        if num_workers is None:
            num_cpu = multiprocessing.cpu_count()
            num_workers = min(num_cpu - 1, len(self.curr_dataloader))
            num_workers = max(num_workers, 0)
        display("num workers for each dataloader is {}".format(num_workers))

        self.loader = torch.utils.data.DataLoader(
            self.curr_dataloader,
            batch_size=1,
            shuffle=False,
            num_workers=num_workers,
            collate_fn=regular_collate,
            timeout=0,
        )

        self.background_rank = background_rank
        self.frame_constant = 1024
        self._initialize_all_normalizers()
        self._initialize_all_background()

    @property
    def order(self):
        return self._order

    def temporal_crop(self, frames):
        """
        Loads a set of frames from the input dataset, without doing any further normalization
        Input:
            frames: a list of frame values (for e.g. [1,5,2,7,8])
        Returns:
            A (potentially motion-corrected) array containing these frames from the tiff dataset with shape (d1, d2, T)
            where d1, d2 are FOV dimensions, T is number of frames selected
        """
        return self.dataset[frames].astype(self.dtype).transpose(1, 2, 0)

    def _initialize_all_normalizers(self):
        """
        Constructs mean image and normalization image
        """
        display("Computing Video Statistics")
        results = self._calculate_mean_and_normalizer()
        self.mean_img = results[0]
        self.std_img = results[1]
        return self.mean_img, self.std_img

    def _initialize_all_background(self):
        self.spatial_basis = self._calculate_background_filter()

    def _calculate_mean_and_normalizer(self, min_allowed_frames: int = 256):
        """
        This function takes a full pass through the dataset and calculates the mean and noise variance at the
        same time
        """
        normalizer_flag = self._compute_normalizer
        if normalizer_flag:
            display("We are normalizing each pixel by a noise variance estimate")
        else:
            display("We are not normalizing each pixel by a noise variance estimate")
        if self.shape[0] < min_allowed_frames:
            normalizer_flag = False

        display("Calculating mean and noise variance")
        overall_mean = np.zeros((self.shape[1], self.shape[2]), dtype=self.dtype)

        if normalizer_flag:
            overall_normalizer = np.zeros(
                (self.shape[1], self.shape[2]), dtype=self.dtype
            )
        else:
            overall_normalizer = np.ones(
                (self.shape[1], self.shape[2]), dtype=self.dtype
            )

        divisor = math.ceil(math.sqrt(self.pixel_batch_size))
        if self.shape[1] - divisor <= 0:
            dim1_range_start_pts = np.arange(1)
        else:
            dim1_range_start_pts = np.arange(0, self.shape[1] - divisor, divisor)
            dim1_range_start_pts = np.concatenate(
                [dim1_range_start_pts, [self.shape[1] - divisor]], axis=0
            )

        if self.shape[2] - divisor <= 0:
            dim2_range_start_pts = np.arange(1)
        else:
            dim2_range_start_pts = np.arange(0, self.shape[2] - divisor, divisor)
            dim2_range_start_pts = np.concatenate(
                [dim2_range_start_pts, [self.shape[2] - divisor]], axis=0
            )

        elts_used = list(range(0, self.shape[0], self.frame_constant))

        elts_for_var_est = 0
        for i in elts_used:
            start_pt_frame = i
            end_pt_frame = min(i + self.frame_constant, self.shape[0])
            data = np.array(
                self.temporal_crop([i for i in range(start_pt_frame, end_pt_frame)])
            )

            data = np.array(data)
            mean_value_net = np.zeros((self.shape[1], self.shape[2]))
            normalizer_net = np.zeros((self.shape[1], self.shape[2]))
            if data.shape[2] >= min_allowed_frames:
                elts_for_var_est += 1
            for step1 in dim1_range_start_pts:
                for step2 in dim2_range_start_pts:
                    crop_data = data[
                        step1 : step1 + divisor, step2 : step2 + divisor, :
                    ]
                    if crop_data.shape[2] >= min_allowed_frames and normalizer_flag:
                        mean_value, noise_est_2d = get_mean_and_noise(
                            crop_data, self.shape[0]
                        )
                        mean_value_net[
                            step1 : step1 + divisor, step2 : step2 + divisor
                        ] = np.array(mean_value)
                        normalizer_net[
                            step1 : step1 + divisor, step2 : step2 + divisor
                        ] = np.array(noise_est_2d)

                    else:
                        mean_value = get_mean_chunk(crop_data, self.shape[0])
                        mean_value_net[
                            step1 : step1 + divisor, step2 : step2 + divisor
                        ] = np.array(mean_value)

            overall_mean += mean_value_net

            if normalizer_flag:
                overall_normalizer += normalizer_net / len(elts_used)

        if normalizer_flag and elts_for_var_est != 0:
            overall_normalizer *= len(elts_used) / elts_for_var_est
            overall_normalizer[overall_normalizer == 0] = 1
        display("Finished mean and noise variance")
        return overall_mean, overall_normalizer

    def temporal_crop_standardized(self, frames):
        crop_data = self.temporal_crop(frames)
        crop_data -= self.mean_img[:, :, None]
        crop_data /= self.std_img[:, :, None]

        return crop_data.astype(self.dtype)

    def _calculate_background_filter(self, n_samples=1000):
        if self.background_rank <= 0:
            return np.zeros((self.shape[1] * self.shape[2], 1)).astype(self.dtype)
        sample_list = [i for i in range(0, self.shape[0])]
        random_data = np.random.choice(
            sample_list, replace=False, size=min(n_samples, self.shape[0])
        ).tolist()
        crop_data = self.temporal_crop_standardized(random_data)
        key = make_jax_random_key()
        spatial_basis, _ = truncated_random_svd(
            crop_data.reshape((-1, crop_data.shape[-1]), order=self.order),
            key,
            self.background_rank,
        )
        return np.array(spatial_basis).astype(self.dtype)

    def v_projection(self, u: coo_matrix, spatial_mixing_matrix: np.ndarray):
        """
        Routine that projects the data onto the spatial matrix.
        Args:
            u (scipy.sparse.coo_matrix): Shape (d, R) where d is number of pixels, R is rank.
                Spatial basis of decomposition
            spatial_mixing_matrix (np.ndarray): U @ spatial_mixing_matrix gives a matrix with orthonormal columns

        Returns:
            (np.ndarray): The result of projecting the whole centered and normalized data onto the spatial basis u
        """
        sparse_projection_term = BCOO.from_scipy_sparse(u.T)
        dense_projection_term = spatial_mixing_matrix.T
        mean_img_r = self.mean_img.reshape((-1, 1), order=self.order)
        std_img_r = self.std_img.reshape((-1, 1), order=self.order)

        result_list = []
        for i, data in enumerate(tqdm(self.loader), 0):
            output = v_projection_routine(
                self.order,
                dense_projection_term,
                sparse_projection_term,
                data,
                mean_img_r,
                std_img_r,
            )

            result_list.append(output)
        result = np.array(jnp.concatenate(result_list, axis=1))

        return result

    def temporal_crop_with_filter(self, frames):
        crop_data = self.temporal_crop(frames)
        spatial_basis_r = self.spatial_basis.reshape(
            (self.shape[1], self.shape[2], -1), order=self.order
        )

        output_matrix = np.zeros(
            (crop_data.shape[0], crop_data.shape[1], crop_data.shape[2])
        )
        temporal_basis = np.zeros((spatial_basis_r.shape[2], crop_data.shape[2]))
        num_iters = math.ceil(output_matrix.shape[2] / self.batch_size)
        start = 0
        for k in range(num_iters):
            end_pt = min(crop_data.shape[2], start + self.batch_size)
            crop_data_subset = crop_data[:, :, start:end_pt]
            filter_data, temporal_basis_crop = standardize_and_filter(
                crop_data_subset, self.mean_img, self.std_img, spatial_basis_r
            )
            filter_data = np.array(filter_data)
            temporal_basis_crop = np.array(temporal_basis_crop)
            output_matrix[:, :, start:end_pt] = filter_data
            temporal_basis[:, start:end_pt] = temporal_basis_crop
            start += self.batch_size
        return output_matrix, temporal_basis


@partial(jit)
def standardize_and_filter(new_data, mean_img, std_img, spatial_basis):
    new_data -= jnp.expand_dims(mean_img, 2)
    new_data /= jnp.expand_dims(std_img, 2)

    d1, d2, t = new_data.shape

    new_data = jnp.reshape(new_data, (d1 * d2, new_data.shape[2]), order="F")
    spatial_basis = jnp.reshape(
        spatial_basis, (d1 * d2, spatial_basis.shape[2]), order="F"
    )

    temporal_projection = jnp.matmul(spatial_basis.T, new_data)
    new_data = new_data - jnp.matmul(spatial_basis, temporal_projection)

    return jnp.reshape(new_data, (d1, d2, t), order="F"), temporal_projection


@partial(jit, static_argnums=(0))
def v_projection_routine(
    order, dense_projection_term, sparse_projection_term, data, mean_img_r, std_img_r
):
    data = jnp.reshape(data, (-1, data.shape[2]), order=order)
    centered_data = (data - mean_img_r) / std_img_r
    output = v_projection_inner_loop(
        dense_projection_term, sparse_projection_term, centered_data
    )
    return output


# @sparse.sparsify
def v_projection_inner_loop(
    dense_projector: ArrayLike, sparse_projector: ArrayLike, data: ArrayLike
) -> Array:
    """
    Inner loop of V projection step
    """
    output = sparse_projector @ data
    output = dense_projector @ output

    return output
