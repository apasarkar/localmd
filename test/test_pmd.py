import tempfile
from pathlib import Path

import numpy as np
import pytest
import tifffile

from localmd.decomposition import localmd_decomposition

def construct_rank_data(rank, dims=(150, 150)):
    spatial_info = np.random.rand(dims[0], dims[1], rank)
    temporal_info = np.random.rand(rank, 1000)
    output = np.tensordot(spatial_info, temporal_info, axes = (2, 0))
    return output.transpose(2, 0, 1)
class TestCompression:

    def setup_method(self):
        self.dataset = construct_rank_data(30)
        self.pmd_params_dict = pmd_params_dict = {
                                    'block_height':32,
                                    'block_width':32,
                                    'frames_to_init':5000,
                                    'background_rank':1,
                                    'max_consec_failures':1,
                                    'max_components':40,
                                    'rank_prune': False,
                                }


    @pytest.mark.parametrize("blocks", [(32, 32), (28, 28), (40, 40)])
    def test_decomposition_blocks(self, blocks):

        block_height = blocks[0]
        block_width = blocks[1]

        block_sizes = [block_height, block_width]

        rank_prune = self.pmd_params_dict['rank_prune']
        max_consec_failures = self.pmd_params_dict['max_consec_failures']
        frames_to_init = self.pmd_params_dict['frames_to_init']
        background_rank = self.pmd_params_dict['background_rank']

        ###THESE PARAMS ARE NEVER MODIFIED
        sim_conf = 5

        max_components = self.pmd_params_dict['max_components']

        registration_routine = None

        frame_batch_size = 2000
        pixel_batch_size = 10000
        dtype = "float32"

        current_video = localmd_decomposition(self.dataset, block_sizes, frames_to_init,\
                                              max_components=max_components, background_rank=background_rank,
                                              sim_conf=sim_conf,\
                                              frame_batch_size=frame_batch_size, pixel_batch_size=pixel_batch_size,
                                              dtype=dtype,\
                                              num_workers=0, registration_routine=registration_routine,
                                              max_consec_failures=max_consec_failures,
                                              rank_prune=rank_prune)

    def teardown_method(self):
        pass