import numpy as np
import pytest

from localmd.decomposition import localmd_decomposition


def construct_rank_data(rank, dims):
    spatial_info = np.random.rand(dims[0], dims[1], rank)
    temporal_info = np.random.rand(rank, 1000)
    output = np.tensordot(spatial_info, temporal_info, axes=(2, 0))
    return output.transpose(2, 0, 1)


class TestCompression:

    def setup_method(self):
        rank = 30
        dims = (150, 150)
        self.dataset = construct_rank_data(rank, dims)
        self.pmd_params_dict = pmd_params_dict = {
            'block_height': 32,
            'block_width': 32,
            'frames_to_init': 5000,
            'background_rank': 1,
            'max_consecutive_failures': 1,
            'max_components': 40,
            'rank_prune': False,
        }

    @pytest.mark.parametrize("block0", [1, 32, 28, 40])
    @pytest.mark.parametrize("block1", [1, 32, 28, 40])
    def test_decomposition_blocks(self, block0, block1):
        if block0 < 10 or block1 < 10:
            pytest.xfail("The block values are too small, this needs to raise an error")
        block_height = block0
        block_width = block1

        block_sizes = [block_height, block_width]

        rank_prune = self.pmd_params_dict['rank_prune']
        max_consecutive_failures = self.pmd_params_dict['max_consecutive_failures']
        frames_to_init = self.pmd_params_dict['frames_to_init']
        background_rank = self.pmd_params_dict['background_rank']

        ###THESE PARAMS ARE NEVER MODIFIED
        sim_conf = 5

        max_components = self.pmd_params_dict['max_components']

        frame_batch_size = 2000
        pixel_batch_size = 10000
        dtype = "float32"

        _ = localmd_decomposition(self.dataset, block_sizes, frames_to_init, \
                                  max_components=max_components, background_rank=background_rank,
                                  sim_conf=sim_conf, \
                                  frame_batch_size=frame_batch_size, pixel_batch_size=pixel_batch_size,
                                  dtype=dtype, \
                                  num_workers=0, \
                                  max_consecutive_failures=max_consecutive_failures,
                                  rank_prune=rank_prune)

    @pytest.mark.parametrize("fov1", [10, 30, 50])
    @pytest.mark.parametrize("fov2", [10, 30, 60])
    @pytest.mark.parametrize("block_sizes", [(32, 32)])
    def test_subselecting_blocks(self, fov1, fov2, block_sizes):
        new_dataset = self.dataset[:, 0:fov1, 0:fov2]

        rank_prune = self.pmd_params_dict['rank_prune']
        max_consecutive_failures = self.pmd_params_dict['max_consecutive_failures']
        frames_to_init = self.pmd_params_dict['frames_to_init']
        background_rank = self.pmd_params_dict['background_rank']

        ###THESE PARAMS ARE NEVER MODIFIED
        sim_conf = 5

        max_components = self.pmd_params_dict['max_components']

        registration_routine = None

        frame_batch_size = 2000
        pixel_batch_size = 10000
        dtype = "float32"

        _ = localmd_decomposition(self.dataset, block_sizes, frames_to_init, \
                                  max_components=max_components, background_rank=background_rank,
                                  sim_conf=sim_conf, \
                                  frame_batch_size=frame_batch_size, pixel_batch_size=pixel_batch_size,
                                  dtype=dtype, \
                                  num_workers=0, \
                                  max_consecutive_failures=max_consecutive_failures,
                                  rank_prune=rank_prune)

    def teardown_method(self):
        pass
