import scipy
import scipy.sparse
import numpy as np
import functools
from functools import partial
import time
import tifffile


def temporal_crop(dataset, frames, frame_corrector = None, batch_size = 100):
    if frame_corrector is not None:
        frame_length = len(frames) 
        result = np.zeros((dataset.shape[0], dataset.shape[1], frame_length))

        value_points = list(range(0, frame_length, batch_size))
        if value_points[-1] > frame_length - batch_size and frame_length > batch_size:
            value_points[-1] = frame_length - batch_size
        for k in value_points:
            start_point = k
            end_point = min(k + batch_size, frame_length)
            curr_frames = frames[start_point:end_point]
            x = dataset.get_frames(curr_frames).astype("float32").transpose(2,0,1)
            result[:, :, start_point:end_point] = np.array(frame_corrector.register_frames(x)).transpose(1,2,0)
            return result
    else:
        return dataset.get_frames(frames).astype("float32")

def generate_PMD_comparison_triptych(dataset, PMD_movie, frames, dim1_interval, dim2_interval):
    
    original_frames = dataset.get_frames(frames).astype("float32").transpose(2, 0, 1)
    original_frames = original_frames[:, dim1_interval[0]:dim1_interval[1], dim2_interval[0]:dim2_interval[1]]
    
    PMD_cropped = PMD_movie[frames, dim1_interval[0]:dim1_interval[1], dim2_interval[0]:dim2_interval[1]]
    
    difference_movie = original_frames - PMD_cropped
    
    outputs = np.concatenate([original_frames, PMD_cropped, difference_movie], axis = 2)
    
    return outputs
